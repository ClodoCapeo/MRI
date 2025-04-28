"use client"

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import type { AnalyzeVolume } from "@/lib/analyze-parser"

interface EvaluationMetricsProps {
  segmentationVolume: AnalyzeVolume | null
  groundTruthVolume: AnalyzeVolume | null
}

export default function EvaluationMetrics({ segmentationVolume, groundTruthVolume }: EvaluationMetricsProps) {
  const [metrics, setMetrics] = useState<{
    dice: number | null
    iou: number | null
    precision: number | null
    sensitivity: number | null
    specificity: number | null
  }>({
    dice: null,
    iou: null,
    precision: null,
    sensitivity: null,
    specificity: null,
  })

  // Add a state for feedback
  const [feedback, setFeedback] = useState<string | null>(null)

  useEffect(() => {
    if (!segmentationVolume || !groundTruthVolume) return

    const calculateMetrics = async () => {
      try {
        // Convert volumes to tensors
        const segData = Array.from(segmentationVolume.data)
        const gtData = Array.from(groundTruthVolume.data)

        // Fix class 4 to class 3 in ground truth
        const fixedGtData = gtData.map((val) => (val === 4 ? 3 : val))

        // Calculate metrics for each class
        const numClasses = 4
        let totalDice = 0
        let totalIou = 0
        let totalPrecision = 0
        let totalSensitivity = 0
        let totalSpecificity = 0
        let classCount = 0

        for (let c = 1; c < numClasses; c++) {
          // Skip background class (0)
          // Create binary masks for current class
          const predMask = segData.map((val) => (val === c ? 1 : 0))
          const gtMask = fixedGtData.map((val) => (val === c ? 1 : 0))

          // Calculate metrics
          let truePositives = 0
          let falsePositives = 0
          let trueNegatives = 0
          let falseNegatives = 0

          for (let i = 0; i < predMask.length; i++) {
            if (predMask[i] === 1 && gtMask[i] === 1) truePositives++
            if (predMask[i] === 1 && gtMask[i] === 0) falsePositives++
            if (predMask[i] === 0 && gtMask[i] === 0) trueNegatives++
            if (predMask[i] === 0 && gtMask[i] === 1) falseNegatives++
          }

          // Only include classes that are present in ground truth
          const gtSum = truePositives + falseNegatives
          if (gtSum > 0) {
            // Dice coefficient: 2*TP / (2*TP + FP + FN)
            const dice = (2 * truePositives) / (2 * truePositives + falsePositives + falseNegatives)

            // IoU: TP / (TP + FP + FN)
            const iou = truePositives / (truePositives + falsePositives + falseNegatives)

            // Precision: TP / (TP + FP)
            const precision = truePositives / (truePositives + falsePositives)

            // Sensitivity (Recall): TP / (TP + FN)
            const sensitivity = truePositives / (truePositives + falseNegatives)

            // Specificity: TN / (TN + FP)
            const specificity = trueNegatives / (trueNegatives + falsePositives)

            totalDice += dice
            totalIou += iou
            totalPrecision += precision
            totalSensitivity += sensitivity
            totalSpecificity += specificity
            classCount++
          }
        }

        // Calculate average metrics
        if (classCount > 0) {
          setMetrics({
            dice: totalDice / classCount,
            iou: totalIou / classCount,
            precision: totalPrecision / classCount,
            sensitivity: totalSensitivity / classCount,
            specificity: totalSpecificity / classCount,
          })
        }
      } catch (error) {
        console.error("Error calculating metrics:", error)
      }
    }

    calculateMetrics()

    // Update the useEffect to generate feedback
    // Add this to the end of the useEffect, right before the closing brace
    if (metrics.dice !== null) {
      setFeedback(generateFeedback(metrics.dice))
    }
  }, [segmentationVolume, groundTruthVolume, metrics.dice])

  // Add a function to generate feedback based on Dice score
  // Add this function after the useEffect
  const generateFeedback = (diceScore: number) => {
    if (diceScore >= 0.85) {
      return "Excellent segmentation! The model is performing very well."
    } else if (diceScore >= 0.75) {
      return "Good segmentation. For further improvement, consider adding more training data or data augmentation."
    } else if (diceScore >= 0.65) {
      return "Moderate segmentation. Consider increasing training epochs, adding data augmentation, or adjusting the learning rate."
    } else {
      return "Poor segmentation. Try the following improvements: 1) Add more training data, 2) Implement data augmentation (rotation, flipping, intensity variations), 3) Adjust model architecture (deeper network or skip connections), 4) Use transfer learning from a pre-trained model."
    }
  }

  if (!segmentationVolume || !groundTruthVolume) return null

  return (
    <Card>
      <CardContent className="pt-6">
        <h3 className="text-lg font-medium mb-4">Evaluation Metrics</h3>

        <div className="space-y-4">
          {/* Dice Coefficient */}
          <div>
            <div className="flex items-center justify-between">
              <span>Dice Coefficient:</span>
              <span className="font-bold">
                {metrics.dice !== null ? `${(metrics.dice * 100).toFixed(2)}%` : "Calculating..."}
              </span>
            </div>
            {metrics.dice !== null && (
              <Progress
                value={metrics.dice * 100}
                className="h-2 mt-2"
                indicatorClassName={`${metrics.dice > 0.8 ? "bg-green-500" : metrics.dice > 0.6 ? "bg-yellow-500" : "bg-red-500"}`}
              />
            )}
          </div>

          {/* IoU */}
          <div>
            <div className="flex items-center justify-between">
              <span>IoU (Intersection over Union):</span>
              <span className="font-bold">
                {metrics.iou !== null ? `${(metrics.iou * 100).toFixed(2)}%` : "Calculating..."}
              </span>
            </div>
            {metrics.iou !== null && (
              <Progress
                value={metrics.iou * 100}
                className="h-2 mt-2"
                indicatorClassName={`${metrics.iou > 0.7 ? "bg-green-500" : metrics.iou > 0.5 ? "bg-yellow-500" : "bg-red-500"}`}
              />
            )}
          </div>

          {/* Precision */}
          <div>
            <div className="flex items-center justify-between">
              <span>Precision:</span>
              <span className="font-bold">
                {metrics.precision !== null ? `${(metrics.precision * 100).toFixed(2)}%` : "Calculating..."}
              </span>
            </div>
            {metrics.precision !== null && (
              <Progress
                value={metrics.precision * 100}
                className="h-2 mt-2"
                indicatorClassName={`${metrics.precision > 0.8 ? "bg-green-500" : metrics.precision > 0.6 ? "bg-yellow-500" : "bg-red-500"}`}
              />
            )}
          </div>

          {/* Sensitivity */}
          <div>
            <div className="flex items-center justify-between">
              <span>Sensitivity (Recall):</span>
              <span className="font-bold">
                {metrics.sensitivity !== null ? `${(metrics.sensitivity * 100).toFixed(2)}%` : "Calculating..."}
              </span>
            </div>
            {metrics.sensitivity !== null && (
              <Progress
                value={metrics.sensitivity * 100}
                className="h-2 mt-2"
                indicatorClassName={`${metrics.sensitivity > 0.8 ? "bg-green-500" : metrics.sensitivity > 0.6 ? "bg-yellow-500" : "bg-red-500"}`}
              />
            )}
          </div>

          {/* Specificity */}
          <div>
            <div className="flex items-center justify-between">
              <span>Specificity:</span>
              <span className="font-bold">
                {metrics.specificity !== null ? `${(metrics.specificity * 100).toFixed(2)}%` : "Calculating..."}
              </span>
            </div>
            {metrics.specificity !== null && (
              <Progress
                value={metrics.specificity * 100}
                className="h-2 mt-2"
                indicatorClassName={`${metrics.specificity > 0.8 ? "bg-green-500" : metrics.specificity > 0.6 ? "bg-yellow-500" : "bg-red-500"}`}
              />
            )}
          </div>
        </div>
        {/* Add the feedback section to the UI */}
        {feedback && (
          <div className="mt-6 p-4 bg-gray-50 rounded-md border">
            <h4 className="font-medium mb-2">Feedback</h4>
            <p className="text-sm">{feedback}</p>
            <p className="text-sm mt-2">
              If you need to improve the model, you can return to the training tab and adjust parameters or add more
              data.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
