import torch
import torch.nn as nn

class NavierStokesLoss(torch.nn.Module):
    def __init__(self, c_physics, mean, std):
        super().__init__()
        self.c_physics = c_physics
        self.mean = mean
        self.std = std

        self.mse = nn.MSELoss()

    def forward(self, input, pred, target):
        data_loss = self.mse(pred, target)
        physics_loss = self.physics_loss(input, pred)

        loss = data_loss + self.c_physics * physics_loss

        return loss, data_loss, physics_loss

    def physics_loss(self, input, pred):
        re = input[:, 1] * self.std['re'] + self.mean['re']

        u = pred[:, 0] * self.std['U_x'] + self.mean['U_x']
        v = pred[:, 1] * self.std['U_y'] + self.mean['U_y']
        p = pred[:, 2] * self.std['p'] + self.mean['p']

        u_t, u_x, u_y, u_xx, u_yy, v_t, v_x, v_y, v_xx, v_yy, p_x, p_y = self.calc_grads(input, u, v, p)

        f_c = u_x + v_y
        f_u = u_t + u * u_x + v * u_y + p_x - 1 / re * (u_xx + u_yy)
        f_v = v_t + u * v_x + v * v_y + p_y - 1 / re * (v_xx + v_yy)

        return torch.mean(f_c ** 2) + torch.mean(f_u ** 2) + torch.mean(f_v ** 2)

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

        u_t = u_grad[:, 0] / self.std['time']
        u_x = u_grad[:, 2] / self.std['x']
        u_y = u_grad[:, 3] / self.std['y']

        v_t = v_grad[:, 0] / self.std['time']
        v_x = v_grad[:, 2] / self.std['x']
        v_y = v_grad[:, 3] / self.std['y']

        p_x = p_grad[:, 2] / self.std['x']
        p_y = p_grad[:, 3] / self.std['y']

        u_xx = torch.autograd.grad(
            u_x,
            input,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0][:, 2] / self.std['x']

        u_yy = torch.autograd.grad(
            u_y,
            input,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True
        )[0][:, 3] / self.std['y']

        v_xx = torch.autograd.grad(
            v_x,
            input,
            grad_outputs=torch.ones_like(v_x),
            create_graph=True
        )[0][:, 2] / self.std['x']

        v_yy = torch.autograd.grad(
            v_y,
            input,
            grad_outputs=torch.ones_like(v_y),
            create_graph=True
        )[0][:, 3] / self.std['y']

        return u_t, u_x, u_y, u_xx, u_yy, v_t, v_x, v_y, v_xx, v_yy, p_x, p_y