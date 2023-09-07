from shap_e.models.download import load_model


# Runs during container build time to get model weights built into the container
def download_model():
    xm = load_model('transmitter')
    model = load_model('text300M')


if __name__ == "__main__":
    download_model()
