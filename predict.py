import torch
from PIL import Image
from torchvision import transforms

from model import MultimodalHousePriceModel


# ----------------------------------------------------
# Device
# ----------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ----------------------------------------------------
# Image Transform
# ----------------------------------------------------

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])


# ----------------------------------------------------
# Load Model
# ----------------------------------------------------

model = MultimodalHousePriceModel(
    tabular_in_dim=3,
    img_feat_dim=256,
    tab_feat_dim=64,
    freeze_backbone=False
)

model.load_state_dict(
    torch.load(
        "best_model.pt",
        map_location=device
    )
)

model.to(device)

model.eval()


# ----------------------------------------------------
# Prediction Function
# ----------------------------------------------------

def predict_price(image, bed, bath, sqft):

    # ---------- Image ----------

    image = image.convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)


    # ---------- Tabular ----------

    tabular = torch.tensor(

        [[bed, bath, sqft]],

        dtype=torch.float32

    ).to(device)


    # ---------- Prediction ----------

    with torch.no_grad():

        prediction = model(

            image,

            tabular

        )


    return prediction.item()
