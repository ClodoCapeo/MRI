import MriSegmentation from "@/components/mri-segmentation"

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center p-8">
      <h1 className="text-3xl font-bold mb-2">U-Net MRI Segmentation</h1>
      <p className="text-gray-500 mb-8">For iSeg 2017 Contest Dataset</p>
      <MriSegmentation />
    </main>
  )
}
