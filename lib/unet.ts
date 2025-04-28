import * as tf from "@tensorflow/tfjs"

export class UNet {
  private model: tf.LayersModel | null = null
  private readonly inputShape: [number, number, number, number] = [null, 256, 256, 2] // [batch, height, width, channels]
  private readonly numClasses: number = 4 // Background + 3 tissue classes (CSF, GM, WM)

  constructor() {}

  async initialize(): Promise<void> {
    try {
      // Make sure TensorFlow.js is ready
      await tf.ready()

      // Try to load a pre-trained model if available
      this.model = await tf.loadLayersModel("indexeddb://unet-iseg-model")
      console.log("Loaded model from IndexedDB")
    } catch (error) {
      console.log("Creating new model...")
      this.model = this.buildUNetModel()

      // Compile the model
      this.model.compile({
        optimizer: tf.train.adam(0.0001),
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
      })

      console.log("Model created and compiled")
    }
  }

  // Update the buildUNetModel method to handle variable input sizes

  private buildUNetModel(): tf.LayersModel {
    // Use null for dynamic dimensions to accept any input size
    const inputs = tf.input({ shape: [null, null, 2] })

    // Encoder (Contracting Path)
    const conv1 = this.convBlock(inputs, 64)
    const pool1 = tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }).apply(conv1) as tf.SymbolicTensor

    const conv2 = this.convBlock(pool1, 128)
    const pool2 = tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }).apply(conv2) as tf.SymbolicTensor

    const conv3 = this.convBlock(pool2, 256)
    const pool3 = tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }).apply(conv3) as tf.SymbolicTensor

    const conv4 = this.convBlock(pool3, 512)
    const pool4 = tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }).apply(conv4) as tf.SymbolicTensor

    // Bridge
    const bridge = this.convBlock(pool4, 1024)

    // Decoder (Expansive Path)
    const upconv1 = tf.layers
      .conv2dTranspose({
        filters: 512,
        kernelSize: [2, 2],
        strides: [2, 2],
        padding: "same",
      })
      .apply(bridge) as tf.SymbolicTensor

    const concat1 = tf.layers.concatenate().apply([upconv1, conv4]) as tf.SymbolicTensor
    const deconv1 = this.convBlock(concat1, 512)

    const upconv2 = tf.layers
      .conv2dTranspose({
        filters: 256,
        kernelSize: [2, 2],
        strides: [2, 2],
        padding: "same",
      })
      .apply(deconv1) as tf.SymbolicTensor

    const concat2 = tf.layers.concatenate().apply([upconv2, conv3]) as tf.SymbolicTensor
    const deconv2 = this.convBlock(concat2, 256)

    const upconv3 = tf.layers
      .conv2dTranspose({
        filters: 128,
        kernelSize: [2, 2],
        strides: [2, 2],
        padding: "same",
      })
      .apply(deconv2) as tf.SymbolicTensor

    const concat3 = tf.layers.concatenate().apply([upconv3, conv2]) as tf.SymbolicTensor
    const deconv3 = this.convBlock(concat3, 128)

    const upconv4 = tf.layers
      .conv2dTranspose({
        filters: 64,
        kernelSize: [2, 2],
        strides: [2, 2],
        padding: "same",
      })
      .apply(deconv3) as tf.SymbolicTensor

    const concat4 = tf.layers.concatenate().apply([upconv4, conv1]) as tf.SymbolicTensor
    const deconv4 = this.convBlock(concat4, 64)

    // Output layer
    const outputs = tf.layers
      .conv2d({
        filters: this.numClasses,
        kernelSize: [1, 1],
        activation: "softmax",
        padding: "same",
      })
      .apply(deconv4) as tf.SymbolicTensor

    return tf.model({ inputs, outputs })
  }

  private convBlock(inputs: tf.SymbolicTensor, filters: number): tf.SymbolicTensor {
    const conv = tf.layers
      .conv2d({
        filters,
        kernelSize: [3, 3],
        activation: "relu",
        padding: "same",
        kernelInitializer: "heNormal",
      })
      .apply(inputs) as tf.SymbolicTensor

    return tf.layers
      .conv2d({
        filters,
        kernelSize: [3, 3],
        activation: "relu",
        padding: "same",
        kernelInitializer: "heNormal",
      })
      .apply(conv) as tf.SymbolicTensor
  }

  async predict(t1Volume: tf.Tensor3D, t2Volume: tf.Tensor3D): Promise<tf.Tensor3D> {
    if (!this.model) {
      throw new Error("Model not initialized")
    }

    return tf.tidy(() => {
      // Get volume dimensions
      const [width, height, depth] = t1Volume.shape

      // Process each slice and combine results
      const resultSlices: tf.Tensor3D[] = []

      for (let z = 0; z < depth; z++) {
        // Extract 2D slices from the volumes
        const t1Slice = t1Volume.slice([0, 0, z], [width, height, 1]).reshape([width, height])
        const t2Slice = t2Volume.slice([0, 0, z], [width, height, 1]).reshape([width, height])

        // Resize slices to 256x256 if needed
        const resizedT1 =
          width !== 256 || height !== 256
            ? tf.image.resizeBilinear(t1Slice.expandDims(2), [256, 256]).squeeze()
            : t1Slice

        const resizedT2 =
          width !== 256 || height !== 256
            ? tf.image.resizeBilinear(t2Slice.expandDims(2), [256, 256]).squeeze()
            : t2Slice

        // Combine slices into a single input tensor
        const combinedSlice = tf.stack([resizedT1, resizedT2], 2).expandDims(0)

        // Run prediction
        const prediction = this.model.predict(combinedSlice) as tf.Tensor4D

        // Get class with highest probability for each pixel
        const segmentation = tf.argMax(prediction, 3).squeeze() as tf.Tensor2D

        // Resize back to original dimensions if needed
        const resizedSegmentation =
          width !== 256 || height !== 256
            ? tf.image.resizeBilinear(segmentation.expandDims(2), [width, height], true).squeeze()
            : segmentation

        // Convert back to integer classes
        const finalSegmentation = tf.cast(tf.round(resizedSegmentation), "int32")

        // Add to result slices
        resultSlices.push(finalSegmentation.expandDims(2))
      }

      // Concatenate all slices along the z-axis
      return tf.concat(resultSlices, 2)
    })
  }

  // Method to process a volume slice by slice
  async processVolume(t1Volume: tf.Tensor3D, t2Volume: tf.Tensor3D): Promise<tf.Tensor3D> {
    if (!this.model) {
      throw new Error("Model not initialized")
    }

    // Get volume dimensions
    const [width, height, depth] = t1Volume.shape

    // Create a tensor to store the result
    const result = tf.buffer([width, height, depth], "int32")

    // Process each slice
    for (let z = 0; z < depth; z++) {
      // Extract 2D slices from the volumes
      const t1Slice = await t1Volume.slice([0, 0, z], [width, height, 1]).reshape([width, height]).array()
      const t2Slice = await t2Volume.slice([0, 0, z], [width, height, 1]).reshape([width, height]).array()

      // Convert to tensors
      const t1SliceTensor = tf.tensor2d(t1Slice)
      const t2SliceTensor = tf.tensor2d(t2Slice)

      // Resize slices to 256x256 if needed
      const resizedT1 =
        width !== 256 || height !== 256
          ? tf.image.resizeBilinear(t1SliceTensor.expandDims(2), [256, 256]).squeeze()
          : t1SliceTensor

      const resizedT2 =
        width !== 256 || height !== 256
          ? tf.image.resizeBilinear(t2SliceTensor.expandDims(2), [256, 256]).squeeze()
          : t2SliceTensor

      // Combine slices into a single input tensor
      const combinedSlice = tf.stack([resizedT1, resizedT2], 2).expandDims(0)

      // Run prediction
      const prediction = this.model.predict(combinedSlice) as tf.Tensor4D

      // Get class with highest probability for each pixel
      const segmentation = tf.argMax(prediction, 3).squeeze() as tf.Tensor2D

      // Resize back to original dimensions if needed
      const resizedSegmentation =
        width !== 256 || height !== 256
          ? tf.image.resizeBilinear(segmentation.expandDims(2), [width, height], true).squeeze()
          : segmentation

      // Get the data and round to integer classes
      const segData = await tf.cast(tf.round(resizedSegmentation), "int32").array()

      // Store in result buffer
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          result.set(segData[y][x], x, y, z)
        }
      }

      // Clean up tensors
      t1SliceTensor.dispose()
      t2SliceTensor.dispose()
      resizedT1.dispose()
      resizedT2.dispose()
      combinedSlice.dispose()
      prediction.dispose()
      segmentation.dispose()
      resizedSegmentation.dispose()
    }

    return result.toTensor()
  }

  async saveModel(path = "downloads://unet-iseg-model"): Promise<void> {
    if (!this.model) {
      throw new Error("Model not initialized")
    }

    await this.model.save(path)
  }

  async loadModel(path: string): Promise<void> {
    this.model = await tf.loadLayersModel(path)

    // Compile the model
    this.model.compile({
      optimizer: tf.train.adam(0.0001),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    })
  }
}
