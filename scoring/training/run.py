#old code i used to first look at the problem. Probably can reuse some for comparision later.
import os
import torch
import torchvision
import pytorch_lightning as pl
from dataset.mnist_no_pil import NoPILMNIST, NoPILMNISTRotationWrapper
import matplotlib.pyplot as plt
from safetensors import safe_open
from safetensors.torch import save_file


class ResNetBlock(torch.nn.Module):
	def __init__(self, channels, kernel_size=3):
		super(ResNetBlock, self).__init__()
		self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size, stride=1, padding="same")
		self.norm1 = torch.nn.BatchNorm2d(channels)
		self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size, stride=1, padding="same")
		self.norm2 = torch.nn.BatchNorm2d(channels)
		self.gelu = torch.nn.GELU()

	def forward(self, x):
		identity = x
		out = self.conv1(x)
		out = self.norm1(out)
		out = self.gelu(out)
		out = self.conv2(out)
		out = self.norm2(out)
		out += identity  # residual
		out = self.gelu(out)
		return out


class Classifier(pl.LightningModule):
	def __init__(self, torchmodel, optimizer_class, optimizer_params, lr_scheduler=None, lr_scheduler_params=None, lr_config=None):
		super(Classifier, self).__init__()
		self.model = torchmodel
		self.loss_fn = torch.nn.CrossEntropyLoss()
		self.optimizer_class = optimizer_class
		self.optimizer_params = optimizer_params
		self.lr_scheduler = lr_scheduler
		self.lr_scheduler_params = lr_scheduler_params
		self.lr_config = lr_config

	def forward(self, x):
		return self.model(x)

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = self.loss_fn(y_hat, y)
		self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = self.loss_fn(y_hat, y)
		accuracy = (y_hat.argmax(dim=1) == y).float().mean()
		self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
		self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
		return loss

	def configure_optimizers(self):
		opt = self.optimizer_class(self.parameters(), **self.optimizer_params)
		if self.lr_scheduler is not None:
			scheduler = self.lr_scheduler(opt, **self.lr_scheduler_params)
			if self.lr_config is not None:
				self.lr_config["scheduler"] = scheduler
			else:
				self.lr_config = {"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss", "frequency": 1, "strict": True, "name": None}

			return {"optimizer": opt, "lr_scheduler": self.lr_config}
		else:
			return opt


class RotationPredictor(pl.LightningModule):
	def __init__(self, torchmodel, optimizer_class, optimizer_params, lr_scheduler=None, lr_scheduler_params=None, lr_config=None):
		super(RotationPredictor, self).__init__()
		self.model = torchmodel
		self.optimizer_class = optimizer_class
		self.optimizer_params = optimizer_params
		self.lr_scheduler = lr_scheduler
		self.lr_scheduler_params = lr_scheduler_params
		self.lr_config = lr_config

	def forward(self, x):
		return self.model(x)

	def unit_circle_loss(self, y_hat, y):
		sin_y = torch.sin(y)
		cos_y = torch.cos(y)
		sin_y_hat = y_hat[:, [1]]
		cos_y_hat = y_hat[:, [0]]
		loss = (sin_y - sin_y_hat) ** 2 + (cos_y - cos_y_hat) ** 2
		return loss.mean()

	def predict_angle(self, x):
		y_hat = self(x)
		angle = torch.atan2(y_hat[:, 1], y_hat[:, 0])
		return angle

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = self.unit_circle_loss(y_hat, y)
		self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = self.unit_circle_loss(y_hat, y)
		self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
		return loss

	def configure_optimizers(self):
		opt = self.optimizer_class(self.parameters(), **self.optimizer_params)
		if self.lr_scheduler is not None:
			scheduler = self.lr_scheduler(opt, **self.lr_scheduler_params)
			if self.lr_config is not None:
				self.lr_config["scheduler"] = scheduler
			else:
				self.lr_config = {"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss", "frequency": 1, "strict": True, "name": None}

			return {"optimizer": opt, "lr_scheduler": self.lr_config}
		else:
			return opt


class DifferentiableRotation(torch.nn.Module):
	def __init__(self):
		super(DifferentiableRotation, self).__init__()

	def forward(self, x, angle):
		# Ensure angle is a tensor with requires_grad=True
		cos_theta = torch.cos(angle)
		sin_theta = torch.sin(angle)
		zero = torch.zeros_like(angle)

		# Construct each row of the rotation matrix
		row1 = torch.stack([cos_theta, -sin_theta, zero])
		row2 = torch.stack([sin_theta, cos_theta, zero])

		# Combine rows into a 2x3 matrix and add batch dimension
		theta = torch.stack([row1, row2]).unsqueeze(0)  # Shape: (1, 2, 3)

		# Generate grid and sample
		grid = torch.nn.functional.affine_grid(theta, x.size(), align_corners=False)
		return torch.nn.functional.grid_sample(x, grid, align_corners=False)

def optimize_rotation(img, model, steps=100, lr=1e-1):
	angle = torch.tensor(0.01, requires_grad=True, device=img.device)
	optimizer = torch.optim.Adam([angle], lr=lr)
	rotation = DifferentiableRotation()
	confidence_history = []


	for _ in range(steps):
		optimizer.zero_grad()
		rotated_img = rotation(img, angle)
		pred = model(rotated_img)
		confidence = torch.nn.functional.softmax(pred, dim=1)
		#use differtiable max here which is
		confidence = torch.sum(confidence.pow(15), dim=1)

		confidence_history.append(confidence.detach().cpu().numpy())



		loss = -confidence  # We want to maximize confidence
		loss.backward()
		optimizer.step()

		with torch.no_grad():
			angle.data = (angle.data + torch.pi) % (2 * torch.pi) - torch.pi

	print(confidence_history)

	#plt.plot(confidence_history)
	#plt.title("Confidence over iterations")
	#plt.xlabel("Iteration")
	#plt.ylabel("Confidence")
	#plt.show()
	return angle,rotated_img

if __name__ == '__main__':
	# Initialize the bot
	#create mnist encoder and decoder that uses resnets


	cls = torch.nn.Sequential(
		torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
		torch.nn.BatchNorm2d(32),
		torch.nn.GELU(),
		ResNetBlock(32),
		torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
		torch.nn.BatchNorm2d(64),
		torch.nn.GELU(),
		ResNetBlock(64),
		torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
		torch.nn.BatchNorm2d(128),
		torch.nn.GELU(),
		ResNetBlock(128),
		torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
		torch.nn.BatchNorm2d(256),
		torch.nn.GELU(),
		torch.nn.Flatten(),
		torch.nn.Linear(256 * 2 * 2, 10),
	)
	random_input = torch.randn(1, 1, 28, 28)
	random_output = cls(random_input)

	cls_model = Classifier(cls, torch.optim.Adam, {"lr": 1e-3}, torch.optim.lr_scheduler.StepLR, {"step_size": 10, "gamma": 0.1})


	#transform to tensor and to range 0-1 Noramlize does not work as we want to have range 0-1
	transform = torchvision.transforms.Compose([
		torchvision.transforms.Lambda(lambda x: x / 255.0),  # Normalize to range [0, 1]
	])


	train = NoPILMNIST(root='data', train=True, download=True, transform=transform)
	val = NoPILMNIST(root='data', train=False, download=True, transform=transform)
	loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val, batch_size=32, shuffle=False)

	if os.path.exists("model.safetensors"):
		with safe_open("model.safetensors", framework="pt", device="cpu") as f:
			state_dict = {key: f.get_tensor(key) for key in f.keys()}
			cls_model.load_state_dict(state_dict)

	else:
		# Create a PyTorch Lightning trainer
		trainer = pl.Trainer(max_epochs=5, accelerator="cuda")
		# Train the model
		trainer.fit(cls_model, loader, val_loader)
		save_file(cls_model.state_dict(), "model.safetensors")



	# Plot some images and their classification with confidence
	plt.figure(figsize=(12, 6))
	for i in range(10):
		img, label = train[i]
		img = img.unsqueeze(0)
		pred = cls_model(img)
		pred_label = pred.argmax(dim=1).item()
		confidence = torch.nn.functional.softmax(pred, dim=1).max().item()
		plt.subplot(2, 5, i + 1)
		plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')
		plt.title(f'T:{label.item()}, P:{pred_label}, C:{confidence:.2f}')
		plt.axis('off')
	plt.tight_layout()
	plt.show()

	# Now randomly rotate image between 45 to 270 degrees and plot with confidence
	angles = torch.randint(45, 270, (10,)).tolist()
	plt.figure(figsize=(12, 6))
	for i in range(10):
		img, label = train[i]
		img = img.unsqueeze(0)
		# Rotate image
		angle = angles[i]
		img = torchvision.transforms.functional.rotate(img, angle)
		pred = cls_model(img)
		pred_label = pred.argmax(dim=1).item()
		confidence = torch.nn.functional.softmax(pred, dim=1).max().item()
		plt.subplot(2, 5, i + 1)
		plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')
		plt.title(f'T:{label.item()}, P:{pred_label}, C:{confidence:.2f}')
		plt.axis('off')
	plt.tight_layout()
	plt.show()

	rotation_model = torch.nn.Sequential(
		torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
		torch.nn.BatchNorm2d(32),
		torch.nn.GELU(),
		ResNetBlock(32),
		torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
		torch.nn.BatchNorm2d(64),
		torch.nn.GELU(),
		ResNetBlock(64),
		torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
		torch.nn.BatchNorm2d(128),
		torch.nn.GELU(),
		ResNetBlock(128),
		torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
		torch.nn.BatchNorm2d(256),
		torch.nn.GELU(),
		torch.nn.Flatten(),
		torch.nn.Linear(256 * 2 * 2, 2),  # Output 2 values representing the complex number (cos, sin)
	)
	random_input = torch.randn(1, 1, 28, 28)
	random_output = rotation_model(random_input)

	rotation_predictor = RotationPredictor(rotation_model, torch.optim.Adam, {"lr": 1e-3}, torch.optim.lr_scheduler.StepLR, {"step_size": 10, "gamma": 0.1})

	if os.path.exists("rotation_model.safetensors"):
		with safe_open("rotation_model.safetensors", framework="pt", device="cpu") as f:
			state_dict = {key: f.get_tensor(key) for key in f.keys()}
			rotation_predictor.load_state_dict(state_dict)
	else:
		# Create a PyTorch Lightning trainer
		trainer = pl.Trainer(max_epochs=5, accelerator="cuda")
		rotated_dataset = NoPILMNISTRotationWrapper(train)
		val_rotated_dataset = NoPILMNISTRotationWrapper(val)

		loader_rotated = torch.utils.data.DataLoader(rotated_dataset, batch_size=32, shuffle=True)
		val_loader_rotated = torch.utils.data.DataLoader(val_rotated_dataset, batch_size=32, shuffle=False)

		# Train the model
		trainer.fit(rotation_predictor, loader_rotated, val_loader_rotated)
		save_file(rotation_predictor.state_dict(), "rotation_model.safetensors")


	plt.figure(figsize=(12, 6))
	for i in range(10):
		img, label = train[i]
		img = img.unsqueeze(0)
		# Rotate image
		angle = angles[i]
		img = torchvision.transforms.functional.rotate(img, angle)

		pred_angle = rotation_predictor.predict_angle(img)
		# Reverse rotate based on pred
		corrected_img = torchvision.transforms.functional.rotate(img, -pred_angle.item() * 180 / torch.pi)
		pred = cls_model(corrected_img)
		pred_label = pred.argmax(dim=1).item()
		confidence = torch.nn.functional.softmax(pred, dim=1).max().item()
		plt.subplot(2, 5, i + 1)
		plt.imshow(corrected_img.squeeze().cpu().numpy(), cmap='gray')
		plt.title(f'T:{label.item()}, P:{pred_label}, C:{confidence:.2f}')
		plt.axis('off')
	plt.tight_layout()
	plt.show()

	plt.figure(figsize=(12, 6))
	for i in range(10):
		img, label = train[i]
		img = img.unsqueeze(0).to(cls_model.device)

		# Rotate image
		angle = angles[i]
		img = torchvision.transforms.functional.rotate(img, angle)

		# Optimize rotation angle to maximize confidence
		optimized_angle,corrected_img = optimize_rotation(img, cls_model)

		pred = cls_model(corrected_img)

		pred_label = pred.argmax(dim=1).item()
		confidence = torch.nn.functional.softmax(pred, dim=1).max().item()

		plt.subplot(2, 5, i + 1)
		plt.imshow(corrected_img.detach().squeeze().cpu().numpy(), cmap='gray')
		plt.title(f'T:{label.item()}, P:{pred_label}, C:{confidence:.2f}')
		plt.axis('off')
	plt.tight_layout()
	plt.show()