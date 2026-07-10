from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import knn_indices, make_uniform_grid, mlp


class PINN(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_size, 256),
            nn.Tanh(),

            nn.Linear(256, 256),
            nn.Tanh(),

            nn.Linear(256, 256),
            nn.Tanh(),

            nn.Linear(256, 256),
            nn.Tanh(),

            nn.Linear(256, out_size),
        )

    def forward(self, x):
        return self.layers(x)
    

################################################################
#                                                              #
#        Geometrically Informed Neural Operators (GINO)        #
#                                                              #                                                           
################################################################


class MeshGNN(nn.Module):
    """
    EdgeConv-stil message passing na k-NN grafu mesh-a.
    Radi na (detachovanim) koordinatama, pa glatkost aktivacije nije bitna ovdje.
    """

    def __init__(self, width, k=8, n_layers=2):
        super().__init__()
        self.k = k
        self.edge_mlps = nn.ModuleList(
            [mlp([2 * width + 2, width, width]) for _ in range(n_layers)]
        )

    def forward(self, pos, feat):
        # pos: (N, 2)  feat: (N, C)
        idx = knn_indices(pos, pos, self.k)         # (N, k)
        for edge_mlp in self.edge_mlps:
            nbr_feat = feat[idx]                     # (N, k, C)
            nbr_rel = pos[idx] - pos.unsqueeze(1)    # (N, k, 2)
            center = feat.unsqueeze(1).expand_as(nbr_feat)
            edge_in = torch.cat([center, nbr_feat - center, nbr_rel], dim=-1)
            msg = edge_mlp(edge_in)                  # (N, k, C)
            feat = feat + msg.mean(dim=1)            # rezidualni update
        return feat
    
class GNOEncoder(nn.Module):
    def __init__(self, in_width, out_width, k=8):
        super().__init__()
        self.k = k
        self.kernel = mlp([in_width + 2, out_width, out_width])

    def forward(self, node_pos, node_feat, grid_pos):
        # node_pos (N,2), node_feat (N,C_in), grid_pos (G,2) -> (G, C_out)
        idx = knn_indices(grid_pos, node_pos, self.k)     # (G, k)
        rel = node_pos[idx] - grid_pos.unsqueeze(1)       # (G, k, 2)
        nf = node_feat[idx]                               # (G, k, C_in)
        msg = self.kernel(torch.cat([nf, rel], dim=-1))   # (G, k, C_out)
        return msg.mean(dim=1)                            # (G, C_out)
    
class GNODecoder(nn.Module):
    def __init__(self, in_width, out_width, k=8):
        super().__init__()
        self.k = k
        # Tanh -> glatko, potrebni drugi izvodi (u_xx, u_yy) za physics loss.
        self.kernel = mlp([in_width + 2, out_width, out_width], act=nn.Tanh, last_act=True)

    def forward(self, query_pos, grid_pos, grid_feat):
        # query_pos (Q,2) [diferencijabilno], grid_pos (G,2), grid_feat (G,C_in)
        idx = knn_indices(query_pos, grid_pos, self.k)      # (Q, k) - nediff selekcija
        rel = query_pos.unsqueeze(1) - grid_pos[idx]        # (Q, k, 2) DIFF po query_pos
        gf = grid_feat[idx]                                 # (Q, k, C_in)
        msg = self.kernel(torch.cat([gf, rel], dim=-1))     # (Q, k, C_out)
        return msg.mean(dim=1)                              # (Q, C_out)


class SpectralConv2d(nn.Module):
    def __init__(self, in_c, out_c, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_c * out_c)
        self.w1 = nn.Parameter(scale * torch.rand(in_c, out_c, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_c, out_c, modes1, modes2, dtype=torch.cfloat))

    @staticmethod
    def _mul(x, w):
        # x: (B, in_c, m1, m2)  w: (in_c, out_c, m1, m2) -> (B, out_c, m1, m2)
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        m1, m2 = min(self.modes1, H // 2), min(self.modes2, W // 2 + 1)
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.w1.shape[1], H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m1, :m2] = self._mul(x_ft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self._mul(x_ft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
    
class FNO2d(nn.Module):
    def __init__(self, width, modes1, modes2, n_layers=4):
        super().__init__()
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)]
        )
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])

    def forward(self, x):
        # x: (C, H, W) -> (C, H, W)
        x = x.unsqueeze(0)
        for i, (spec, loc) in enumerate(zip(self.spectral, self.local)):
            y = spec(x) + loc(x)
            x = F.gelu(y) if i < len(self.spectral) - 1 else y
        return x.squeeze(0)


class GINO(nn.Module):
    def __init__(
        self,
        in_size=4,
        out_size=3,
        width=32,
        latent_size=32,
        fno_modes=12,
        fno_layers=4,
        gno_k=8,
        use_gnn=True,
        gnn_k=8,
        gnn_layers=2,
    ):
        super().__init__()
        self.latent_size = latent_size

        self.node_lift = mlp([in_size, width, width], act=nn.GELU, last_act=True)
        self.use_gnn = use_gnn
        if use_gnn:
            self.gnn = MeshGNN(width, k=gnn_k, n_layers=gnn_layers)

        self.encoder = GNOEncoder(width, width, k=gno_k)
        self.fno = FNO2d(width, fno_modes, fno_modes, n_layers=fno_layers)
        self.decoder = GNODecoder(width, width, k=gno_k)
        self.head = mlp([width, width, out_size], act=nn.Tanh)

    @classmethod
    def from_config(cls, config):
        """
        Gradi GINO iz config-a: putanja do YAML fajla, dict, ili dict sa 'gino' blokom
        (npr. ucitan config/architecture/gino.yaml).
        """
        if isinstance(config, (str, Path)):
            with open(config) as f:
                config = yaml.safe_load(f)
        cfg = config.get("gino", config) if isinstance(config, dict) else config
        return cls(**cfg)

    def forward(self, input):
        # input: (B, 4) = [time, re, x, y].  requires_grad postavlja trening petlja.
        query_pos = input[:, 2:4]                 # ZIVE koordinate (diff putanja ka izlazu)

        ctx = input.detach()                      # kontekst za enkoder: bez grad ka ulazu
        node_pos = ctx[:, 2:4]
        node_feat = self.node_lift(ctx)           # (B, width)
        if self.use_gnn:
            node_feat = self.gnn(node_pos, node_feat)

        grid_pos = make_uniform_grid(node_pos, self.latent_size)   # (G, 2)
        latent = self.encoder(node_pos, node_feat, grid_pos)       # (G, width)

        S = self.latent_size
        grid = latent.t().reshape(1, -1, S, S).squeeze(0)          # (width, S, S)
        grid = self.fno(grid)                                      # (width, S, S)
        grid_feat = grid.reshape(grid.shape[0], -1).t()            # (G, width)

        dec = self.decoder(query_pos, grid_pos, grid_feat)         # (B, width)
        return self.head(dec)                                      # (B, out_size)