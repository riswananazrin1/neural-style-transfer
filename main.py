import tensorflow as tf
import numpy as np
from PIL import Image
import cv2




IMAGE_SIZE = 256
CONTENT_WEIGHT = 1e4
STYLE_WEIGHT = 1e-2
TOTAL_VARIATION_WEIGHT = 30
STEPS = 300



def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image = np.array(image)
    image = image.astype(np.float32)

    image = np.expand_dims(image, axis=0)

    return image


def preprocess_image(image):
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return image




print("Loading VGG-19...")

vgg = tf.keras.applications.VGG19(
    include_top=False,
    weights="imagenet"
)

vgg.trainable = False

print("VGG-19 loaded successfully.")




content_layer = "block5_conv2"

style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1"
]

selected_layers = style_layers + [content_layer]

outputs = [vgg.get_layer(layer).output for layer in selected_layers]

feature_extractor = tf.keras.Model(
    inputs=vgg.input,
    outputs=outputs
)


def gram_matrix(input_tensor):
    result = tf.linalg.einsum(
        "bijc,bijd->bcd",
        input_tensor,
        input_tensor
    )

    input_shape = tf.shape(input_tensor)

    num_locations = tf.cast(
        input_shape[1] * input_shape[2],
        tf.float32
    )

    return result / num_locations




def get_features(image):
    image = preprocess_image(image)

    outputs = feature_extractor(image)

    style_outputs = outputs[:len(style_layers)]
    content_output = outputs[len(style_layers)]

    style_features = [
        gram_matrix(style_output)
        for style_output in style_outputs
    ]

    content_features = content_output

    return style_features, content_features




content_path = "content_img/content.jpg"
style_path = "style_img/style.jpg"

print("Loading images...")

content_image = load_image(content_path)
style_image = load_image(style_path)

print("Images loaded successfully.")




style_targets, _ = get_features(style_image)

_, content_target = get_features(content_image)




generated_image = tf.Variable(
    content_image,
    dtype=tf.float32
)




def calculate_loss(generated_image):
    style_outputs, content_output = get_features(generated_image)

    style_loss = 0

    for generated_style, target_style in zip(
        style_outputs,
        style_targets
    ):
        style_loss += tf.reduce_mean(
            tf.square(generated_style - target_style)
        )

    style_loss /= len(style_layers)

    content_loss = tf.reduce_mean(
        tf.square(content_output - content_target)
    )

    total_loss = (
        CONTENT_WEIGHT * content_loss
        + STYLE_WEIGHT * style_loss
        + TOTAL_VARIATION_WEIGHT
        * tf.reduce_mean(
            tf.image.total_variation(generated_image)
        )
    )

    return total_loss




optimizer = tf.optimizers.Adam(
    learning_rate=1.0,
    beta_1=0.99,
    epsilon=1e-1
)


@tf.function
def train_step(image):
    with tf.GradientTape() as tape:

        loss = calculate_loss(image)

    gradients = tape.gradient(
        loss,
        image
    )

    optimizer.apply_gradients(
        [(gradients, image)]
    )

    image.assign(
        tf.clip_by_value(
            image,
            0,
            255
        )
    )

    return loss



print("Starting Neural Style Transfer...")

for step in range(STEPS):

    loss = train_step(generated_image)

    if (step + 1) % 50 == 0:
        print(
            f"Step {step + 1}/{STEPS} - "
            f"Loss: {loss.numpy():.2f}"
        )




stylized_image = generated_image.numpy()[0]

stylized_image = np.clip(
    stylized_image,
    0,
    255
).astype(np.uint8)

Image.fromarray(stylized_image).save(
    "outputs/stylized_result.jpg"
)

print("Stylized image saved successfully.")




def create_sketch(image_path, output_path):

    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    inverted_image = cv2.bitwise_not(
        gray_image
    )

    blurred_image = cv2.GaussianBlur(
        inverted_image,
        (21, 21),
        0
    )

    sketch = cv2.divide(
        gray_image,
        255 - blurred_image,
        scale=256
    )

    cv2.imwrite(
        output_path,
        sketch
    )



print("Creating sketch...")

create_sketch(
    content_path,
    "outputs/sketch_result.jpg"
)

print("Sketch saved successfully.")



print("\nProject completed!")
print("NST result: outputs/stylized_result.jpg")
print("Sketch result: outputs/sketch_result.jpg")