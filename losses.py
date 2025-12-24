import torch 
import torch.autograd.functional as func


def combined_loss(model, source_samples, target_samples, t, r, epoch):
    
    velocity = source_samples - target_samples
    x_interp = (1 - t) * target_samples + t * source_samples
    tau = torch.clamp(t - r, min=1e-8)

    def f(x, t_val, r_val):
        return model(x, t_val, r_val)

    tangents_full = (
        velocity,
        torch.ones_like(t),
        torch.zeros_like(r)
    )



    tangents_partial_x = (
        velocity,
        torch.zeros_like(t),
        torch.zeros_like(r)
    )

    # Compute u and derivatives
    u, dudt = func.jvp(f, (x_interp, t, r), tangents_full, create_graph=True)


    _, nabla_x_u_dot_v = func.jvp(f, (x_interp, t, r), tangents_partial_x, create_graph=True)


    target_1 = velocity - tau * (dudt)
    loss_1 = torch.nn.functional.mse_loss(u, target_1)

    target_2 = velocity - u  - tau * (dudt - nabla_x_u_dot_v)
    pred_scale = (tau * nabla_x_u_dot_v).norm(dim=-1)
    true_scale = target_2.norm(dim=-1)

    loss_2 = torch.nn.functional.l1_loss(pred_scale, true_scale)
    return    0.3 * loss_1 + loss_2

def backward_loss(model, source_samples, target_samples, t, r):

    x_interp = (1 - t) * target_samples + t * source_samples
    velocity = source_samples - target_samples

    def f(x, t_val, r_val):
        return model(x, t_val, r_val)

    tangents = (
        velocity,                  # dx = velocity (full dim)
        torch.ones_like(t),        # dt = 1
        torch.zeros_like(r)        # dr = 0
    )

    u, dudt = func.jvp(f, (x_interp, t, r), tangents , create_graph=True)  # ← REQUIR)
    

    u_target = velocity - (t - r) * dudt

    loss =  torch.nn.functional.mse_loss(u, u_target.detach())


    return loss

