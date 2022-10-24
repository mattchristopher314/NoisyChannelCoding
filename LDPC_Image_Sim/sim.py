import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageOps
from utils import greyscale_to_binary, binary_to_greyscale
from ldpc import initialise_ldpc, encode_image, decode_image, extract_message_bits
import math
import copy
import random

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

img = Image.open("EssayTalk.png")
img = ImageOps.exif_transpose(img)

img = np.asarray(img.convert('LA'))[:, :, 0]
Image.fromarray(img).save("Greyscale.png")

img_binary = greyscale_to_binary(img)

n = 16
columnDensity = 3
rowDensity = 4

H, G = initialise_ldpc(n, rowDensity, columnDensity)

noiseVariance = 0.25

img_encoded = encode_image(G, img_binary)
encodingNoise = np.random.randn(
    *img_encoded.shape)*math.sqrt(noiseVariance)

img_correct = img_encoded[:].T

img_encoded += encodingNoise

print("Beginning decoding attempt -", end="")
img_decoded, convergenceData = decode_image(
    G, H, noiseVariance, copy.deepcopy(img_encoded), img_binary.shape)
print(" done.")

f, axes = plt.subplots(1, 3, figsize=(18, 6))
axes = axes.flatten()

extracted = []
for block in range(img_encoded.shape[1]):
    extracted.append(extract_message_bits(
        G.T, (img_encoded[:, block] < 0).astype(int)))
extracted = np.array(extracted).T

noisyOut = binary_to_greyscale(np.array(extracted).flatten()[:np.prod(
    img_binary.shape)].reshape(*img_binary.shape) % 2)
decodedOut = binary_to_greyscale(np.array(img_decoded).flatten()[:np.prod(
    img_binary.shape)].reshape(*img_binary.shape) % 2)

Image.fromarray(noisyOut.astype(np.uint8)).save("Noisy.png")
Image.fromarray(decodedOut.astype(np.uint8)).save("Decoded.png")

noisyBER = 0
finalBER = 0

flatBinary = img_binary.flatten()
flatNoisy = extracted.flatten()
flatDecoded = img_decoded.flatten()

for i in range(len(flatBinary)):
    if flatBinary[i] != flatNoisy[i]:
        noisyBER += 1
    if flatBinary[i] != flatDecoded[i]:
        finalBER += 1

for title, im, ax in zip(["Original", "Noisy", "Decoded"], [img, binary_to_greyscale(np.array(extracted).flatten()[:np.prod(img_binary.shape)].reshape(*img_binary.shape) % 2), binary_to_greyscale(np.array(img_decoded).flatten()[:np.prod(img_binary.shape)].reshape(*img_binary.shape) % 2)], axes):
    ax.imshow(im, cmap="gray")
    ax.set_title(title, fontsize=30)
    ax.set_xticks([])
    ax.set_yticks([])
    if title == "Decoded":
        ax.set_xlabel(r"Errors: " + str(round(finalBER, 2)), fontsize=30)
    elif title == "Noisy":
        ax.set_xlabel(r"Errors: " + str(round(noisyBER, 2)), fontsize=30)

plt.get_current_fig_manager().set_window_title("Image Simulation Results")
plt.tight_layout()

print("Saving output -", end="")
plt.savefig("ImageSimulationResults.png", dpi=300, bbox_inches="tight")
print(" done.")
# plt.show()

plt.style.use("bmh")
plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.xlim(0, 20)

plt.title("Message Bit Value Convergence over Iterations")
plt.xlabel("Iteration")
plt.ylabel("A Posteriori Message Bit LLR Value")

plt.plot(range(21), convergenceData.T)
plt.savefig("Convergence.png", dpi=300, bbox_inches="tight")
plt.show()
