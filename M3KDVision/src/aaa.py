import torch
max_steps = 930
micro_loss = 10
for global_step in range(1,max_steps):
    beta_t = 0.5 * min(1.0, global_step / max(max_steps, 1))

    safe_scale = beta_t / (1.0 + micro_loss)
    #a = torch.tensor(beta_t*micro_loss)
    #safe_scale = torch.tanh(torch.tensor(beta_t*micro_loss)) 
    micro_loss = beta_t / (1.0 + micro_loss)  * micro_loss
    micro_loss = torch.tanh(torch.tensor(beta_t*micro_loss))  * micro_loss

    beta_t_old = 0.5 * min(1.0, global_step / max(max_steps, 1))
    old_loss = beta_t_old * micro_loss
    print(combined_micro, old_loss)
    
