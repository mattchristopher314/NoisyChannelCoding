import copy
import sys
import random
from matplotlib import pyplot as plt
from PIL import Image, ImageOps


def Parity(data):
    return [data[0] ^ data[1] ^ data[2], data[1] ^ data[2] ^ data[3], data[2] ^ data[3] ^ data[0]]


def Probability(received, transmitted, total, flipProb):
    result = 1

    for i in range(len(received)):
        if int(received[i]) != transmitted[i]:
            result *= flipProb

    return result


f = 0.05

img = Image.open(sys.argv[1]).convert("L")
img = ImageOps.exif_transpose(img)

img.save("GreyscaleOriginal.png")
print("Converted to greyscale.")

pixels = img.load()

blocks = []
decoderBlocks = []

for i in range(img.size[0]):
    for j in range(img.size[1]):
        pixelVal = [int(x) for x in list(format(pixels[i, j], f"#010b")[2:])]
        blocks.append(pixelVal[:4] + Parity(pixelVal[:4]))
        blocks.append(pixelVal[4:] + Parity(pixelVal[4:]))

correct = copy.deepcopy(blocks)
print("Encoded image.")

initial = img.copy()

for i in range(len(blocks)):
    for j in range(4+3):
        if random.uniform(0, 1) < f:
            blocks[i][j] = 0 if blocks[i][j] == 1 else 1
        else:
            blocks[i][j] = 1 if blocks[i][j] == 1 else 0

for i in range(0, len(blocks), 2):
    pixel = "".join([str(x) for x in blocks[i]])[:4] + \
        "".join([str(x) for x in blocks[i+1]])[:4]
    pixels[i/2 // img.size[1], i/2 % img.size[1]] = int(pixel, 2)

errors = 0
for i in range(len(blocks)):
    for j in range(len(blocks[i])):
        if blocks[i][j] != correct[i][j]:
            errors += 1

img.save(f"Noisy(f={f}).png")
noisyBER = errors/(len(blocks)*len(blocks[0]))
print("Simulated transmission by adding noise. Error rate: ",
      noisyBER)

noisy = img.copy()

for i, block in enumerate(blocks):
    probs = []
    vectors = []
    string = "".join([str(x) for x in block])
    for j in range(16):
        data = f"{j:04b}"
        vectors.append([int(x) for x in
                        data + "".join([str(x) for x in Parity([0 if x == "0" else 1 for x in data])])])

        probs.append(Probability(string, vectors[j], 4, f))

    tol = 0.0001
    probThreshold = max(probs)

    for k, codeword in enumerate(vectors):
        if abs(probs[k] - probThreshold) < tol:
            blocks[i] = vectors[k]
            next

for i in range(0, len(blocks), 2):
    pixel = "".join([str(x) for x in blocks[i]])[:4] + \
        "".join([str(x) for x in blocks[i+1]])[:4]
    pixels[i/2 // img.size[1], i/2 %
           img.size[1]] = int(pixel, 2)

errors = 0
for i in range(len(blocks)):
    for j in range(len(blocks[i])):
        if blocks[i][j] != correct[i][j]:
            errors += 1

img.save(f"Decoded(f={f}).png")
finalBER = errors/(len(blocks)*len(blocks[0]))
print("Attempted recovery of source image. Error rate: ",
      finalBER)


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

f, axes = plt.subplots(1, 3, figsize=(18, 6))
axes = axes.flatten()


for title, im, ax in zip(["Original", "Noisy", "Decoded"], [initial, noisy, img], axes):
    ax.imshow(im, cmap="gray")
    ax.set_title(title, fontsize=30)
    ax.set_xticks([])
    ax.set_yticks([])
    if title == "Decoded":
        ax.set_xlabel(r"BER $\approx$ " + str(round(finalBER, 2)), fontsize=30)
    elif title == "Noisy":
        ax.set_xlabel(r"BER $\approx$ " + str(round(noisyBER, 2)), fontsize=30)

plt.get_current_fig_manager().set_window_title("Image Simulation Results")
plt.tight_layout()

print("Saving output -", end="")
plt.savefig("ImageSimulationResults.png", dpi=300, bbox_inches="tight")
print(" done.")
plt.show()
