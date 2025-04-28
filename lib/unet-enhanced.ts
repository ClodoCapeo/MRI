import * as tf from "@tensorflow/tfjs"

export class EnhancedUNet {
  private model: tf.LayersModel | null = null
  public modelStep1: tf.LayersModel | null = null
  public modelStep2: tf.LayersModel | null = null
  public modelStep3: tf.LayersModel | null = null
  private readonly inputShape: [number, number, number] = [128, 128, 2] // Default: 2 channels (T1+T2)
  private readonly numClasses: number = 4 // Multi-class segmentation (4 classes)
  public modelPath = "indexeddb://unet-iseg-model" // Default model path
  public modelPathStep1 = "indexeddb://unet-iseg-model-step1"
  public modelPathStep2 = "indexeddb://unet-iseg-model-step2"
  public modelPathStep3 = "indexeddb://unet-iseg-model-step3"

  constructor(customModelPath?: string) {
    if (customModelPath) {
      this.modelPath = customModelPath
    }
  }

  // Gaussian normalization function
  private gaussianNormalization(image: tf.Tensor): tf.Tensor {
    const positiveValues = image.where(image.greater(0), tf.zerosLike(image))
    const mean = positiveValues.mean()
    const std = positiveValues.std()
    const normalizedImage = image.sub(mean).div(tf.scalar(5).mul(std))
    return normalizedImage
  }

  async initialize(inputChannels = 2): Promise<void> {
    try {
      await tf.ready()

      // Initialize Step 1 model (binary segmentation of brain)
      try {
        this.modelStep1 = await tf.loadLayersModel(this.modelPathStep1)
        console.log(`Loaded model step 1 from ${this.modelPathStep1}`)
      } catch (error) {
        console.log("Creating new model step 1...")
        this.modelStep1 = this.buildUNetModelBinary(inputChannels)

        // Binary classification with binary crossentropy
        this.modelStep1.compile({
          optimizer: tf.train.adam(0.0001),
          loss: "binaryCrossentropy",
          metrics: ["accuracy"],
        })

        console.log("Model step 1 created and compiled")
      }

      // Initialize Step 2 model (binary segmentation of white matter)
      try {
        this.modelStep2 = await tf.loadLayersModel(this.modelPathStep2)
        console.log(`Loaded model step 2 from ${this.modelPathStep2}`)
      } catch (error) {
        console.log("Creating new model step 2...")
        this.modelStep2 = this.buildUNetModelBinary(3) // Step 2 takes 3 channels (T1, T2, Step 1 output)

        this.modelStep2.compile({
          optimizer: tf.train.adam(0.0001),
          loss: "binaryCrossentropy",
          metrics: ["accuracy"],
        })

        console.log("Model step 2 created and compiled")
      }

      // Initialize Step 3 model (multi-class segmentation)
      try {
        this.modelStep3 = await tf.loadLayersModel(this.modelPathStep3)
        console.log(`Loaded model step 3 from ${this.modelPathStep3}`)
      } catch (error) {
        console.log("Creating new model step 3...")
        this.modelStep3 = this.buildUNetModelMultiClass(4) // Step 3 takes 4 channels (T1, T2, Step 1, Step 2 outputs)

        this.modelStep3.compile({
          optimizer: tf.train.adam(0.0001),
          loss: "categoricalCrossentropy",
          metrics: ["accuracy"],
        })

        console.log("Model step 3 created and compiled")
      }
    } catch (error) {
      console.error("Failed to initialize model:", error)
      throw error
    }
  }

  // Build U-Net model for binary segmentation (Step 1 and Step 2)
  private buildUNetModelBinary(inputChannels = 2): tf.LayersModel {
    // Input layer with dynamic channel count
    const inputs = tf.input({ shape: [128, 128, inputChannels] })

    // Encoder (Contracting Path)
    const conv1 = this.convBlock(inputs, 64)
    const pool1 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(conv1) as tf.SymbolicTensor

    const conv2 = this.convBlock(pool1, 128)
    const pool2 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(conv2) as tf.SymbolicTensor

    const conv3 = this.convBlock(pool2, 256)
    const pool3 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(conv3) as tf.SymbolicTensor

    const conv4 = this.convBlock(pool3, 512)
    const pool4 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(conv4) as tf.SymbolicTensor

    // Bridge
    const conv5 = this.convBlock(pool4, 1024)
    const dropout = tf.layers.dropout({ rate: 0.3 }).apply(conv5) as tf.SymbolicTensor

    // Decoder (Expansive Path)
    const up6 = tf.layers.upSampling2d({ size: [2, 2] }).apply(dropout) as tf.SymbolicTensor
    const up6Conv = tf.layers
        .conv2d({
          filters: 512,
          kernelSize: 2,
          activation: "relu",
          padding: "same",
          kernelInitializer: "heNormal",
        })
        .apply(up6) as tf.SymbolicTensor
    const merge6 = tf.layers.concatenate().apply([conv4, up6Conv]) as tf.SymbolicTensor
    const conv6 = this.convBlock(merge6, 512)

    const up7 = tf.layers.upSampling2d({ size: [2, 2] }).apply(conv6) as tf.SymbolicTensor
    const up7Conv = tf.layers
        .conv2d({
          filters: 256,
          kernelSize: 2,
          activation: "relu",
          padding: "same",
          kernelInitializer: "heNormal",
        })
        .apply(up7) as tf.SymbolicTensor
    const merge7 = tf.layers.concatenate().apply([conv3, up7Conv]) as tf.SymbolicTensor
    const conv7 = this.convBlock(merge7, 256)

    const up8 = tf.layers.upSampling2d({ size: [2, 2] }).apply(conv7) as tf.SymbolicTensor
    const up8Conv = tf.layers
        .conv2d({
          filters: 128,
          kernelSize: 2,
          activation: "relu",
          padding: "same",
          kernelInitializer: "heNormal",
        })
        .apply(up8) as tf.SymbolicTensor
    const merge8 = tf.layers.concatenate().apply([conv2, up8Conv]) as tf.SymbolicTensor
    const conv8 = this.convBlock(merge8, 128)

    const up9 = tf.layers.upSampling2d({ size: [2, 2] }).apply(conv8) as tf.SymbolicTensor
    const up9Conv = tf.layers
        .conv2d({
          filters: 64,
          kernelSize: 2,
          activation: "relu",
          padding: "same",
          kernelInitializer: "heNormal",
        })
        .apply(up9) as tf.SymbolicTensor
    const merge9 = tf.layers.concatenate().apply([conv1, up9Conv]) as tf.SymbolicTensor
    const conv9 = this.convBlock(merge9, 64)

    // Output layer with sigmoid activation for binary classification
    const outputs = tf.layers
        .conv2d({
          filters: 1, // Binary output (1 channel)
          kernelSize: 1,
          activation: "sigmoid",
          padding: "same",
        })
        .apply(conv9) as tf.SymbolicTensor

    return tf.model({ inputs, outputs })
  }

  // Build U-Net model for multi-class segmentation (Step 3)
  private buildUNetModelMultiClass(inputChannels = 4): tf.LayersModel {
    // Input layer with dynamic channel count
    const inputs = tf.input({ shape: [128, 128, inputChannels] })

    // Encoder (Contracting Path)
    const conv1 = this.convBlock(inputs, 64)
    const pool1 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(conv1) as tf.SymbolicTensor

    const conv2 = this.convBlock(pool1, 128)
    const pool2 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(conv2) as tf.SymbolicTensor

    const conv3 = this.convBlock(pool2, 256)
    const pool3 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(conv3) as tf.SymbolicTensor

    const conv4 = this.convBlock(pool3, 512)
    const pool4 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(conv4) as tf.SymbolicTensor

    // Bridge
    const conv5 = this.convBlock(pool4, 1024)
    const dropout = tf.layers.dropout({ rate: 0.3 }).apply(conv5) as tf.SymbolicTensor

    // Decoder (Expansive Path)
    const up6 = tf.layers.upSampling2d({ size: [2, 2] }).apply(dropout) as tf.SymbolicTensor
    const up6Conv = tf.layers
        .conv2d({
          filters: 512,
          kernelSize: 2,
          activation: "relu",
          padding: "same",
          kernelInitializer: "heNormal",
        })
        .apply(up6) as tf.SymbolicTensor
    const merge6 = tf.layers.concatenate().apply([conv4, up6Conv]) as tf.SymbolicTensor
    const conv6 = this.convBlock(merge6, 512)

    const up7 = tf.layers.upSampling2d({ size: [2, 2] }).apply(conv6) as tf.SymbolicTensor
    const up7Conv = tf.layers
        .conv2d({
          filters: 256,
          kernelSize: 2,
          activation: "relu",
          padding: "same",
          kernelInitializer: "heNormal",
        })
        .apply(up7) as tf.SymbolicTensor
    const merge7 = tf.layers.concatenate().apply([conv3, up7Conv]) as tf.SymbolicTensor
    const conv7 = this.convBlock(merge7, 256)

    const up8 = tf.layers.upSampling2d({ size: [2, 2] }).apply(conv7) as tf.SymbolicTensor
    const up8Conv = tf.layers
        .conv2d({
          filters: 128,
          kernelSize: 2,
          activation: "relu",
          padding: "same",
          kernelInitializer: "heNormal",
        })
        .apply(up8) as tf.SymbolicTensor
    const merge8 = tf.layers.concatenate().apply([conv2, up8Conv]) as tf.SymbolicTensor
    const conv8 = this.convBlock(merge8, 128)

    const up9 = tf.layers.upSampling2d({ size: [2, 2] }).apply(conv8) as tf.SymbolicTensor
    const up9Conv = tf.layers
        .conv2d({
          filters: 64,
          kernelSize: 2,
          activation: "relu",
          padding: "same",
          kernelInitializer: "heNormal",
        })
        .apply(up9) as tf.SymbolicTensor
    const merge9 = tf.layers.concatenate().apply([conv1, up9Conv]) as tf.SymbolicTensor
    const conv9 = this.convBlock(merge9, 64)

    // Output layer with softmax activation for multi-class segmentation
    const outputs = tf.layers
        .conv2d({
          filters: this.numClasses, // Multi-class output (4 channels)
          kernelSize: 1,
          activation: "softmax",
          padding: "same",
        })
        .apply(conv9) as tf.SymbolicTensor

    return tf.model({ inputs, outputs })
  }

  // Enhanced convBlock with more parameters
  private convBlock(inputs: tf.SymbolicTensor, filters: number): tf.SymbolicTensor {
    const conv = tf.layers
        .conv2d({
          filters,
          kernelSize: 3,
          activation: "relu",
          padding: "same",
          kernelInitializer: "heNormal",
        })
        .apply(inputs) as tf.SymbolicTensor

    // Add batch normalization for better training stability
    const batchNorm = tf.layers.batchNormalization().apply(conv) as tf.SymbolicTensor

    return tf.layers
        .conv2d({
          filters,
          kernelSize: 3,
          activation: "relu",
          padding: "same",
          kernelInitializer: "heNormal",
        })
        .apply(batchNorm) as tf.SymbolicTensor
  }

  // Train Step 1 model (binary segmentation of brain)
  async trainStep1(
      trainXs: tf.Tensor4D,
      trainYs: tf.Tensor4D,
      valXs: tf.Tensor4D,
      valYs: tf.Tensor4D,
      epochs = 100,
      batchSize = 1,
      callbacks: tf.Callback[] = [],
  ): Promise<tf.History> {
    if (!this.modelStep1) {
      throw new Error("Step 1 model not initialized")
    }

    console.log("Starting Step 1 training...")
    console.log("Train shapes:", trainXs.shape, trainYs.shape)
    console.log("Validation shapes:", valXs.shape, valYs.shape)

    // Train the model
    const history = await this.modelStep1.fit(trainXs, trainYs, {
      epochs,
      batchSize,
      validationData: [valXs, valYs],
      callbacks,
      shuffle: true,
    })

    return history
  }

  // Train Step 2 model (binary segmentation of white matter)
  async trainStep2(
      trainXs: tf.Tensor4D,
      trainYs: tf.Tensor4D,
      valXs: tf.Tensor4D,
      valYs: tf.Tensor4D,
      epochs = 100,
      batchSize = 1,
      callbacks: tf.Callback[] = [],
  ): Promise<tf.History> {
    if (!this.modelStep2) {
      throw new Error("Step 2 model not initialized")
    }

    console.log("Starting Step 2 training...")
    console.log("Train shapes:", trainXs.shape, trainYs.shape)
    console.log("Validation shapes:", valXs.shape, valYs.shape)

    // Train the model
    const history = await this.modelStep2.fit(trainXs, trainYs, {
      epochs,
      batchSize,
      validationData: [valXs, valYs],
      callbacks,
      shuffle: true,
    })

    return history
  }

  // Train Step 3 model (multi-class segmentation)
  async trainStep3(
      trainXs: tf.Tensor4D,
      trainYs: tf.Tensor4D,
      valXs: tf.Tensor4D,
      valYs: tf.Tensor4D,
      epochs = 100,
      batchSize = 1,
      callbacks: tf.Callback[] = [],
  ): Promise<tf.History> {
    if (!this.modelStep3) {
      throw new Error("Step 3 model not initialized")
    }

    console.log("Starting Step 3 training...")
    console.log("Train shapes:", trainXs.shape, trainYs.shape)
    console.log("Validation shapes:", valXs.shape, valYs.shape)

    // Train the model
    const history = await this.modelStep3.fit(trainXs, trainYs, {
      epochs,
      batchSize,
      validationData: [valXs, valYs],
      callbacks,
      shuffle: true,
    })

    return history
  }

  // Legacy train method for backward compatibility
  async train(
      trainData: { xs: tf.Tensor4D; ys: tf.Tensor4D }[],
      validationData: { xs: tf.Tensor4D; ys: tf.Tensor4D }[],
      epochs = 100,
      batchSize = 1,
      callbacks: tf.Callback[] = [],
  ): Promise<tf.History> {
    if (!this.modelStep1) {
      throw new Error("Model not initialized")
    }

    if (trainData.length === 0) {
      throw new Error("No training data provided")
    }

    try {
      // Log shapes for debugging
      console.log("Training data shapes:")
      trainData.forEach((item, i) => {
        console.log(`Item ${i}: xs shape=${item.xs.shape}, ys shape=${item.ys.shape}`)
      })

      // Flatten the data arrays
      console.log("Concatenating training data...")
      const trainXs = tf.concat(
          trainData.map((d) => d.xs),
          0,
      )
      const trainYs = tf.concat(
          trainData.map((d) => d.ys),
          0,
      )

      console.log("Training data concatenated. Shapes:", trainXs.shape, trainYs.shape)

      let valXs, valYs
      if (validationData.length > 0) {
        console.log("Concatenating validation data...")
        valXs = tf.concat(
            validationData.map((d) => d.xs),
            0,
        )
        valYs = tf.concat(
            validationData.map((d) => d.ys),
            0,
        )
        console.log("Validation data concatenated. Shapes:", valXs.shape, valYs.shape)
      }

      // Train the model
      console.log("Starting model.fit...")
      const history = await this.modelStep1.fit(trainXs, trainYs, {
        epochs,
        batchSize,
        validationData: valXs && valYs ? [valXs, valYs] : undefined,
        callbacks,
        shuffle: true,
      })

      // Clean up
      trainXs.dispose()
      trainYs.dispose()
      if (valXs) valXs.dispose()
      if (valYs) valYs.dispose()

      return history
    } catch (error) {
      console.error("Error during training:", error)
      throw error
    }
  }

  async predict(inputVolumes: tf.Tensor3D[]): Promise<tf.Tensor3D> {
    // If models aren't initialized and we're just viewing ground truth
    if ((!this.modelStep1 || !this.modelStep2 || !this.modelStep3) && inputVolumes.length === 1) {
      // Just return the input volume (assuming it's the ground truth)
      return inputVolumes[0].clone()
    }

    if (!this.modelStep1 || !this.modelStep2 || !this.modelStep3) {
      throw new Error("Models not initialized")
    }

    if (inputVolumes.length === 0) {
      throw new Error("No input volumes provided")
    }

    return tf.tidy(() => {
      try {
        // Get volume dimensions from the first volume
        const [width, height, depth] = inputVolumes[0].shape

        // Check that all volumes have the same dimensions
        for (let i = 1; i < inputVolumes.length; i++) {
          if (!tf.util.arraysEqual(inputVolumes[0].shape, inputVolumes[i].shape)) {
            throw new Error(
                `Input volumes must have the same dimensions. Volume 0: ${inputVolumes[0].shape}, Volume ${i}: ${inputVolumes[i].shape}`,
            )
          }
        }

        // Process each slice and combine results
        const resultSlices: tf.Tensor3D[] = []

        for (let z = 0; z < depth; z++) {
          // Extract 2D slices from each volume
          const slices = inputVolumes.map((volume) =>
              volume.slice([0, 0, z], [width, height, 1]).reshape([width, height]),
          )

          // Resize slices to 128x128
          const resizedSlices = slices.map((slice) =>
              tf.image.resizeBilinear(slice.expandDims(2), [128, 128]).squeeze(),
          )

          // Combine slices into a single input tensor for Step 1
          const step1Input = tf.stack(resizedSlices.slice(0, 2), 2).expandDims(0) // T1 and T2 only

          // Normalize to [0,1]
          const normalizedStep1Input = step1Input.div(step1Input.max())

          // Run Step 1 prediction (brain segmentation)
          const step1Prediction = this.modelStep1.predict(normalizedStep1Input) as tf.Tensor4D
          const step1Binary = step1Prediction.greater(0.5).toFloat().squeeze(0)

          // Combine T1, T2, and Step 1 prediction for Step 2
          const step2Input = tf
              .concat([resizedSlices[0].expandDims(2), resizedSlices[1].expandDims(2), step1Binary], 2)
              .expandDims(0)

          // Run Step 2 prediction (white matter segmentation)
          const step2Prediction = this.modelStep2.predict(step2Input) as tf.Tensor4D
          const step2Binary = step2Prediction.greater(0.5).toFloat().squeeze(0)

          // Combine T1, T2, Step 1, and Step 2 predictions for Step 3
          const step3Input = tf
              .concat([resizedSlices[0].expandDims(2), resizedSlices[1].expandDims(2), step1Binary, step2Binary], 2)
              .expandDims(0)

          // Run Step 3 prediction (multi-class segmentation)
          const step3Prediction = this.modelStep3.predict(step3Input) as tf.Tensor4D

          // Get class with highest probability for each pixel
          const segmentation = tf.argMax(step3Prediction, 3).squeeze() as tf.Tensor2D

          // Resize back to original dimensions
          const resizedSegmentation = tf.image
              .resizeBilinear(segmentation.expandDims(2), [width, height], true)
              .squeeze()

          // Convert back to integer classes
          const finalSegmentation = tf.cast(tf.round(resizedSegmentation), "int32")

          // Add to result slices
          resultSlices.push(finalSegmentation.expandDims(2))
        }

        // Concatenate all slices along the z-axis
        return tf.concat(resultSlices, 2)
      } catch (error) {
        console.error("Error during prediction:", error)
        throw error
      }
    })
  }

  // Post-processing with argmax decoding as shown in tutorial
  async predictWithPostProcessing(inputVolumes: tf.Tensor3D[]): Promise<tf.Tensor3D> {
    // If models aren't initialized and we're just viewing ground truth
    if ((!this.modelStep1 || !this.modelStep2 || !this.modelStep3) && inputVolumes.length === 1) {
      // Just return the input volume (assuming it's the ground truth)
      return inputVolumes[0].clone()
    }

    // For now, this is the same as predict since we're already doing argmax decoding
    return this.predict(inputVolumes)
  }

  async saveModel(path = "indexeddb://unet-iseg-model"): Promise<tf.io.SaveResult> {
    if (!this.modelStep1) {
      throw new Error("Model not initialized")
    }
    return await this.modelStep1.save(path)
  }

  getModelPath(): string {
    return this.modelPath
  }

  setModelPath(path: string): void {
    this.modelPath = path
  }

  // Dice coefficient calculation
  dice(X: tf.Tensor, Y: tf.Tensor): number {
    const XBIN = X.greater(0)
    const YBIN = Y.greater(0)
    const INTERSECTION = tf.sum(tf.logicalAnd(XBIN, YBIN)).dataSync()[0]
    const XCARD = tf.sum(XBIN).dataSync()[0]
    const YCARD = tf.sum(YBIN).dataSync()[0]
    const diceCoefficient = (2.0 * INTERSECTION) / (XCARD + YCARD)
    return XCARD + YCARD > 0 ? diceCoefficient : -1
  }
}
