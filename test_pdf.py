from reportlab.pdfgen import canvas

pdf = canvas.Canvas("sample.pdf")
pdf.drawString(100,750,"PDF Working")
pdf.save()

print("PDF Created")