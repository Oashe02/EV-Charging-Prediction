import os
import re
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'EV Infrastructure Planning Report', 0, 1, 'C')
        self.line(10, 22, 200, 22)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(markdown_content: str) -> bytes:
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    if os.path.exists("data/report_chart.png"):
        pdf.image("data/report_chart.png", x=15, w=180)
        pdf.ln(5)
        

    for line in markdown_content.split('\n'):

        line = line.encode('latin-1', 'replace').decode('latin-1')
        

        if line.startswith('### '):
            pdf.set_font('Arial', 'B', 12)
            pdf.multi_cell(0, 7, line.replace('### ', '').replace('**', ''))
            pdf.ln(2)
        elif line.startswith('## '):
            pdf.set_font('Arial', 'B', 14)
            pdf.multi_cell(0, 8, line.replace('## ', '').replace('**', ''))
            pdf.ln(3)
        elif line.startswith('# '):
            pdf.set_font('Arial', 'B', 16)
            pdf.multi_cell(0, 10, line.replace('# ', '').replace('**', ''))
            pdf.ln(4)
            

        elif line.strip().startswith('- ') or line.strip().startswith('* '):
            pdf.set_font('Arial', '', 11)
            pdf.multi_cell(0, 6, "   " + chr(149) + " " + line.strip()[2:].replace('**', ''))
            pdf.ln(1)
            
        elif not line.strip():
            pdf.ln(3)
            

        else:
            pdf.set_font('Arial', '', 11)
            pdf.multi_cell(0, 6, line.replace('**', ''))
            pdf.ln(1)
            
    return bytes(pdf.output())
