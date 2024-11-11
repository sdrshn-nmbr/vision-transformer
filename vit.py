import argparse, os, math, requests, logging
from datetime import datetime
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from safetensors.torch import save_file, load_file

os.makedirs("logs", exist_ok=True)

log_filename = datetime.now().strftime("logs/log_%Y%m%d_%H%M%S.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.FileHandler(log_filename, mode="a"), logging.StreamHandler()],
)
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"using device: {device}")


class Encoder(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super(Encoder, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        assert (
            self.head_dim * num_heads == dim
        ), "embedding dimension must be divisible by number of heads"

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, src):
        # normalize the input
        src_norm = self.norm1(src)

        # project and split into q k and v
        q, k, v = rearrange(
            self.qkv_proj(src_norm),
            "b s (three h d) -> three b h s d",
            three=3,
            h=self.num_heads,
            d=self.head_dim,
        )

        # compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v)

        # concatenate heads and project
        attn_output = rearrange(attn_output, "b h s d -> b s (h d)")
        attn_output = self.out_proj(attn_output)
        attn_output = self.out_dropout(attn_output)

        # add residual
        src = src + attn_output

        # feed forward
        src_norm = self.norm2(src)
        mlp_output = self.mlp(src_norm)

        # add residual
        src = src + mlp_output

        return src


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dropout=0.1,
    ):
        super(ViT, self).__init__()
        assert (
            image_size % patch_size == 0
        ), "image size must be divisible by the patch size"
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_size = patch_size
        self.dim = dim

        self.patch_embeddings = nn.Conv2d(
            in_channels=channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))

        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.ModuleList(
            [
                Encoder(dim, heads, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, img):
        # convert image to patches
        x = self.patch_embeddings(img)  # (batch_size, dim, h, w)
        x = rearrange(x, "b c h w -> b (h w) c")  # (batch_size, num_patches, dim)

        b, n, _ = x.shape

        # add cls token
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, dim)
        x = x + self.pos_embedding[:, : n + 1]
        x = self.dropout(x)

        # pass through transformer
        for layer in self.transformer:
            x = layer(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]  # (batch_size, dim)
        logits = self.fc(cls_token_final)
        return logits


def load_pretrained_weights(model, pretrained_state_dict):
    model_state_dict = model.state_dict()
    new_state_dict = {}

    # map the keys
    for key in pretrained_state_dict.keys():
        new_key = key

        # remove 'vit.' prefix if present (dont know why we need this but seems to work)
        if new_key.startswith("vit."):
            new_key = new_key.replace("vit.", "")

        # embeddings
        if new_key == "embeddings.cls_token":
            new_key = "cls_token"
        elif new_key == "embeddings.position_embeddings":
            new_key = "pos_embedding"
        elif new_key == "embeddings.patch_embeddings.projection.weight":
            new_key = "patch_embeddings.weight"
        elif new_key == "embeddings.patch_embeddings.projection.bias":
            new_key = "patch_embeddings.bias"

        # transformer layers
        elif new_key.startswith("encoder.layer"):
            layer_num = int(new_key.split(".")[2])
            sub_key = ".".join(new_key.split(".")[3:])

            # map subcomponents
            if sub_key == "layernorm_before.weight":
                new_key = f"transformer.{layer_num}.norm1.weight"
            elif sub_key == "layernorm_before.bias":
                new_key = f"transformer.{layer_num}.norm1.bias"
            elif sub_key == "attention.attention.query.weight":
                new_key = f"transformer.{layer_num}.qkv_proj.weight"
                q_weight = pretrained_state_dict[key]
                k_weight = pretrained_state_dict[
                    f"vit.encoder.layer.{layer_num}.attention.attention.key.weight"
                ]
                v_weight = pretrained_state_dict[
                    f"vit.encoder.layer.{layer_num}.attention.attention.value.weight"
                ]
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                new_state_dict[new_key] = qkv_weight
                continue
            elif sub_key == "attention.attention.query.bias":
                new_key = f"transformer.{layer_num}.qkv_proj.bias"
                q_bias = pretrained_state_dict[key]
                k_bias = pretrained_state_dict[
                    f"vit.encoder.layer.{layer_num}.attention.attention.key.bias"
                ]
                v_bias = pretrained_state_dict[
                    f"vit.encoder.layer.{layer_num}.attention.attention.value.bias"
                ]
                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                new_state_dict[new_key] = qkv_bias
                continue
            elif sub_key == "attention.output.dense.weight":
                new_key = f"transformer.{layer_num}.out_proj.weight"
            elif sub_key == "attention.output.dense.bias":
                new_key = f"transformer.{layer_num}.out_proj.bias"
            elif sub_key == "layernorm_after.weight":
                new_key = f"transformer.{layer_num}.norm2.weight"
            elif sub_key == "layernorm_after.bias":
                new_key = f"transformer.{layer_num}.norm2.bias"
            elif sub_key == "intermediate.dense.weight":
                new_key = f"transformer.{layer_num}.mlp.0.weight"
            elif sub_key == "intermediate.dense.bias":
                new_key = f"transformer.{layer_num}.mlp.0.bias"
            elif sub_key == "output.dense.weight":
                new_key = f"transformer.{layer_num}.mlp.3.weight"
            elif sub_key == "output.dense.bias":
                new_key = f"transformer.{layer_num}.mlp.3.bias"
            else:
                logger.info(f"skipping {key}")
                continue
        elif new_key == "layernorm.weight":
            new_key = "norm.weight"
        elif new_key == "layernorm.bias":
            new_key = "norm.bias"
        elif new_key == "classifier.weight":
            new_key = "fc.weight"
        elif new_key == "classifier.bias":
            new_key = "fc.bias"
        else:
            logger.info(f"skipping {key}")
            continue

        if new_key in model_state_dict:
            new_state_dict[new_key] = pretrained_state_dict[key]
        else:
            logger.warning(f"{new_key} not in model state dict")

    # update the model state dict
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="vision transformer training and inference"
    )
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument(
        "--use_trained_weights",
        action="store_true",
        help="use trained weights for inference if available",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate the model on the test dataset",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="path to the input image for inference",
    )
    args = parser.parse_args()

    checkpoint_dir = "./checkpoints"
    model_weights_path = os.path.join(checkpoint_dir, "model_weights.safetensors")
    training_metadata_path = os.path.join(checkpoint_dir, "training_metadata.pt")

    # make sure checkpoints directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # check for cuda again just in case
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"using device: {device}")

    if args.train:
        # setup training transforms
        transform_train = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True, num_workers=4
        )

        # initialize model
        model = ViT(
            image_size=224,
            patch_size=16,
            num_classes=10,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            dropout=0.1,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        batch_losses = []
        epoch_losses = []

        # load checkpoint if exists
        start_epoch = 0
        if os.path.exists(model_weights_path) and os.path.exists(
            training_metadata_path
        ):
            logger.info("loading checkpoint")
            # load model weights
            pretrained_state_dict = load_file(model_weights_path, device=device)
            model.load_state_dict(pretrained_state_dict)
            # load training metadata
            checkpoint = torch.load(training_metadata_path, map_location=device)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            logger.info(f"resuming training from epoch {start_epoch}")
        else:
            logger.info("no checkpoint found starting training from scratch")

        # training loop
        num_epochs = 20
        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # track loss
                batch_losses.append(loss.item())
                running_loss += loss.item()

                if i % 100 == 99:
                    logger.info(
                        f"epoch [{epoch+1}/{num_epochs}], step [{i+1}/{len(train_loader)}], loss: {running_loss / 100:.4f}"
                    )
                    running_loss = 0.0

            # calculate epoch loss
            epoch_loss = sum(batch_losses[-len(train_loader) :]) / len(train_loader)
            epoch_losses.append(epoch_loss)
            logger.info(f"epoch [{epoch+1}/{num_epochs}], epoch loss: {epoch_loss:.4f}")

            # save model weights
            save_file(model.state_dict(), model_weights_path)
            logger.info(f"model weights saved at epoch {epoch+1}")

            # save training metadata
            checkpoint_data = {
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint_data, training_metadata_path)
            logger.info(f"training metadata saved at epoch {epoch+1}")

        logger.info("training complete")

        # save loss plots
        os.makedirs("graphs", exist_ok=True)

        # plot batch losses
        plt.figure(figsize=(12, 6))
        plt.plot(batch_losses, label="batch loss")
        plt.xlabel("batch")
        plt.ylabel("loss")
        plt.title("batch loss over training")
        plt.legend()
        plt.savefig("graphs/batch_loss.png")
        plt.close()

        # plot epoch losses
        plt.figure(figsize=(12, 6))
        plt.plot(epoch_losses, label="epoch loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("epoch loss over training")
        plt.legend()
        plt.savefig("graphs/epoch_loss.png")
        plt.close()

    elif args.evaluate:
        # initialize model
        model = ViT(
            image_size=224,
            patch_size=16,
            num_classes=10,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            dropout=0.1,
        ).to(device)

        # load trained weights
        if os.path.exists(model_weights_path):
            logger.info("loading trained weights from checkpoint")
            try:
                # First load to CPU
                pretrained_state_dict = load_file(model_weights_path, device="cpu")
                # Then move to appropriate device
                pretrained_state_dict = {k: v.to(device) for k, v in pretrained_state_dict.items()}
                model.load_state_dict(pretrained_state_dict)
                logger.info("trained weights loaded successfully")
            except Exception as e:
                logger.error(f"Error loading weights: {e}")
                exit(1)
        else:
            logger.error("no trained weights found please train the model first")
            exit(1)

        # setup test transforms
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        test_loader = DataLoader(
            test_dataset, batch_size=64, shuffle=False, num_workers=4
        )

        # evaluate the model
        model.eval()
        correct = 0
        total = 0
        try:
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            logger.info(f"test accuracy: {accuracy:.2f}%")
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            exit(1)

    else:
        # decide number of classes
        if args.use_trained_weights:
            num_classes = 10  # cifar10
        else:
            num_classes = 1000  # imagenet

        # initialize model
        model = ViT(
            image_size=224,
            patch_size=16,
            num_classes=num_classes,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            dropout=0.1,
        ).to(device)

        # load weights
        if args.use_trained_weights:
            if os.path.exists(model_weights_path):
                logger.info("loading trained weights from checkpoint")
                pretrained_state_dict = load_file(model_weights_path, device=device)
                model.load_state_dict(pretrained_state_dict)
                logger.info("trained weights loaded")
            else:
                logger.warning(
                    "no trained weights found using pre-trained weights instead"
                )
                args.use_trained_weights = False  # fallback

        if not args.use_trained_weights:
            # Attempt to download safetensors if available, else fallback to pytorch_model.bin
            repo_id = "google/vit-base-patch16-224"
            safetensors_filename = "model.safetensors"
            pytorch_bin_filename = "pytorch_model.bin"
            try:
                # Try to download safetensors
                state_dict_path = hf_hub_download(
                    repo_id=repo_id, filename=safetensors_filename
                )
                pretrained_state_dict = load_file(state_dict_path, device=device)
                model.load_state_dict(pretrained_state_dict)
                logger.info("pre-trained weights loaded from safetensors")
            except Exception as e:
                logger.warning(
                    f"Could not load safetensors format: {e}. Falling back to pytorch_model.bin"
                )
                # Download pytorch_model.bin and load with mapping
                state_dict_path = hf_hub_download(
                    repo_id=repo_id, filename=pytorch_bin_filename
                )
                pretrained_state_dict = torch.load(state_dict_path, map_location=device)
                load_pretrained_weights(model, pretrained_state_dict)
                logger.info("pre-trained weights loaded from pytorch_model.bin")

        # load image
        if args.image_path is not None:
            if not os.path.exists(args.image_path):
                logger.error(f"image not found at path: {args.image_path}")
                exit(1)
            try:
                img = Image.open(args.image_path).convert("RGB")
                logger.info(f"loaded image from {args.image_path}")
            except Exception as e:
                logger.error(f"error loading image: {e}")
                exit(1)
        else:
            # default image
            url = "https://pytorch.org/assets/images/dog.jpg"
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            logger.info("loaded default image from url")

        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        img = preprocess(img).unsqueeze(0).to(device)

        # do inference
        model.eval()
        with torch.no_grad():
            outputs = model(img)
            probs = F.softmax(outputs, dim=-1)
            top_probs, top_idxs = probs.topk(5)

        if args.use_trained_weights:
            labels = [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ]
            logger.info("top predictions:")
            for prob, idx in zip(top_probs[0], top_idxs[0]):
                label = labels[idx]
                logger.info(f"{label}: {prob.item() * 100:.2f}%")
        else:
            try:
                response = requests.get(
                    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
                )
                response.raise_for_status()
                labels = response.json()
            except Exception as e:
                logger.error(f"Failed to download ImageNet labels: {e}")
                exit(1)
            logger.info("top predictions:")
            for prob, idx in zip(top_probs[0], top_idxs[0]):
                if idx < len(labels):
                    label = labels[idx]
                else:
                    label = f"Class {idx}"
                logger.info(f"{label}: {prob.item() * 100:.2f}%")

## usage
# for training
# python vit.py --train
# for evaluation
# python vit.py --evaluate
# for inference with pre-trained weights
# python vit.py --image_path /path/to/your/image.jpg
# for inference with your trained weights
# python vit.py --use_trained_weights --image_path /path/to/your/image.jpg
