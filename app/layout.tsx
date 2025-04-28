import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'MRI PFEE',
  description: 'Created per Mathias Tock',
  generator: 'MRI PFEE',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
