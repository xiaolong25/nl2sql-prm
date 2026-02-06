import matplotlib.pyplot as plt

def plot_losses(losses, save_path):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(save_path)
    plt.close()
