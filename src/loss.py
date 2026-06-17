import torch
import torch.nn as nn

class NavierStokesLoss(torch.nn.Module):
    def __init__(self, c_physics):
        super().__init__()
        self.c_physics = c_physics

    def forward(self, input, pred, target):
        re = input[:, 1]

        u = pred[:, 0]
        v = pred[:, 1]
        p = pred[:, 2]

        u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y = self.calc_grads(input, u, v, p)

        # data loss
        loss = nn.MSELoss()(pred, target)

        # physics loss
        f_c = u_x + v_y
        f_u = u * u_x + v * u_y + p_x - 1 / re * (u_xx + u_yy)
        f_v = u * v_x + v * v_y + p_y - 1 / re * (v_xx + v_yy)

        loss += self.c_physics * (torch.mean(f_c ** 2) + torch.mean(f_u ** 2) + torch.mean(f_v ** 2))

        return loss

    def calc_grads(self, input, u, v, p):
        u_grad = torch.autograd.grad(
            u,
            input,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        v_grad = torch.autograd.grad(
            v,
            input,
            grad_outputs=torch.ones_like(v),
            create_graph=True
        )[0]

        p_grad = torch.autograd.grad(
            p,
            input,
            grad_outputs=torch.ones_like(p),
            create_graph=True
        )[0]

        u_x = u_grad[:, 2]
        u_y = u_grad[:, 3]

        v_x = v_grad[:, 2]
        v_y = v_grad[:, 3]

        p_x = p_grad[:, 2]
        p_y = p_grad[:, 3]

        u_xx = torch.autograd.grad(
            u_x,
            input,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0][:, 2]

        u_yy = torch.autograd.grad(
            u_y,
            input,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True
        )[0][:, 3]

        v_xx = torch.autograd.grad(
            v_x,
            input,
            grad_outputs=torch.ones_like(v_x),
            create_graph=True
        )[0][:, 2]

        v_yy = torch.autograd.grad(
            v_y,
            input,
            grad_outputs=torch.ones_like(v_y),
            create_graph=True
        )[0][:, 3]

        return u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y
