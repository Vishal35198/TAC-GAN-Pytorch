import torch

with torch.no_grad():
    G.eval()
    test_noise = torch.randn(8, 100, device=device)
    test_text = torch.randn(8, 10, 384, device=device)  # Mock embeddings
    samples = G(test_noise, test_text)
    save_image(samples*0.5+0.5, "debug_sample.png")  # Scale to [0,1]