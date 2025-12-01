import torch

@torch.no_grad()
def meanflow_sampler(
    model, 
    latents,
    cfg_scale=1.0,
    num_steps=1, 
    timesteps=None,
    **kwargs
):
    """
    MeanFlow sampler supporting both single-step and multi-step generation
    """
    batch_size = latents.shape[0]
    device = latents.device
    
    if num_steps == 1 and timesteps is None:
        r = torch.zeros(batch_size, device=device)
        t = torch.ones(batch_size, device=device)
        u = model(latents, noise_labels_r=r, noise_labels_t=t)
        # x_0 = x_1 - u(x_1, 0, 1)
        x0 = latents - u
        
    else:
        z = latents
        if timesteps is None:
            time_steps = torch.linspace(1, 0, num_steps + 1, device=device)
        else:
            # Ensure timesteps are on the correct device
            if not isinstance(timesteps, torch.Tensor):
                time_steps = torch.tensor(timesteps, device=device)
            else:
                time_steps = timesteps.to(device)
                
            # If timesteps are increasing (0->1), flip them to be 1->0 for generation
            if time_steps[0] < time_steps[-1]:
                time_steps = torch.flip(time_steps, dims=[0])
                
            num_steps = len(time_steps) - 1

        for i in range(num_steps):
            t_cur = time_steps[i]
            t_next = time_steps[i + 1]
            
            t = torch.full((batch_size,), t_cur, device=device)
            r = torch.full((batch_size,), t_next, device=device)

            u = model(z, noise_labels_r=r, noise_labels_t=t)
            
            # Update z: z_r = z_t - (t-r)*u(z_t, r, t)
            z = z - (t_cur - t_next) * u
        
        x0 = z
    
    return x0