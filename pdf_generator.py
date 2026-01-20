"""
PDF Report Generation for KYC Verification

Generates verification certificate PDFs that can be downloaded.
Uses reportlab for PDF generation.
"""

import io
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try to import PDF libraries
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
    logger.info("ReportLab PDF library loaded")
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not available - PDF generation disabled")


def generate_verification_pdf(verification_data: Dict[str, Any]) -> Optional[bytes]:
    """
    Generate a PDF verification certificate.
    
    Args:
        verification_data: Dictionary containing verification results
        
    Returns:
        PDF bytes or None if generation fails
    """
    if not REPORTLAB_AVAILABLE:
        logger.warning("Cannot generate PDF - ReportLab not installed")
        return None
    
    try:
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                               rightMargin=20*mm, leftMargin=20*mm,
                               topMargin=20*mm, bottomMargin=20*mm)
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            alignment=TA_CENTER,
            spaceAfter=12*mm,
            textColor=colors.HexColor('#1e40af')
        )
        subtitle_style = ParagraphStyle(
            'SubTitle',
            parent=styles['Normal'],
            fontSize=12,
            alignment=TA_CENTER,
            spaceAfter=6*mm,
            textColor=colors.HexColor('#64748b')
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=4*mm,
            textColor=colors.HexColor('#334155')
        )
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=2*mm
        )
        
        # Build content
        content = []
        
        # Header
        content.append(Paragraph("KYC VERIFICATION CERTIFICATE", title_style))
        content.append(Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M IST')}", 
            subtitle_style
        ))
        content.append(Spacer(1, 10*mm))
        
        # Verification Status
        status = verification_data.get('status', 'unknown')
        status_color = colors.HexColor('#10b981') if status == 'verified' else colors.HexColor('#ef4444')
        status_text = "✓ VERIFIED" if status == 'verified' else "✗ NOT VERIFIED"
        
        status_style = ParagraphStyle(
            'Status',
            parent=styles['Heading1'],
            fontSize=20,
            alignment=TA_CENTER,
            textColor=status_color,
            spaceAfter=10*mm
        )
        content.append(Paragraph(status_text, status_style))
        content.append(Spacer(1, 5*mm))
        
        # Verification ID
        verification_id = verification_data.get('id', datetime.now().strftime('%Y%m%d%H%M%S'))
        content.append(Paragraph(f"Verification ID: {verification_id}", subtitle_style))
        content.append(Spacer(1, 10*mm))
        
        # Document Details
        content.append(Paragraph("Document Details", heading_style))
        
        extracted = verification_data.get('extracted_data', {})
        doc_type = verification_data.get('doc_type', 'Unknown')
        
        # Table data
        table_data = [
            ['Field', 'Value'],
            ['Document Type', doc_type],
            ['Name', extracted.get('Name', 'N/A')],
            ['Date of Birth', extracted.get('DOB', 'N/A')],
            ['Gender', extracted.get('Gender', 'N/A')],
        ]
        
        # Mask Aadhaar number
        aadhaar = extracted.get('Aadhaar Number', '')
        if aadhaar and len(aadhaar) >= 4:
            masked_aadhaar = f"XXXX XXXX {aadhaar[-4:]}"
        else:
            masked_aadhaar = 'N/A'
        table_data.append(['Aadhaar Number', masked_aadhaar])
        
        # Create table
        table = Table(table_data, colWidths=[60*mm, 100*mm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ]))
        content.append(table)
        content.append(Spacer(1, 10*mm))
        
        # Verification Details
        content.append(Paragraph("Verification Details", heading_style))
        
        verification_info = verification_data.get('verification', {})
        match_score = verification_info.get('match_score', 'N/A')
        matched_fields = verification_info.get('matched_fields', [])
        
        verification_table_data = [
            ['Metric', 'Result'],
            ['Document Confidence', f"{verification_data.get('confidence', 0) * 100:.1f}%"],
            ['Database Match Score', f"{match_score}%"],
            ['Fields Matched', ', '.join(matched_fields) if matched_fields else 'N/A'],
            ['Verification Time', datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')],
        ]
        
        verification_table = Table(verification_table_data, colWidths=[60*mm, 100*mm])
        verification_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8b5cf6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ]))
        content.append(verification_table)
        content.append(Spacer(1, 15*mm))
        
        # Footer disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#94a3b8')
        )
        content.append(Paragraph(
            "This certificate is generated by AI KYC Verification System. "
            "It is valid only for the purpose stated and is subject to verification. "
            "Any tampering or misuse of this document is a punishable offense.",
            disclaimer_style
        ))
        
        # Build PDF
        doc.build(content)
        
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
        
    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        return None


def generate_simple_pdf(verification_data: Dict[str, Any]) -> bytes:
    """
    Generate a simple text-based PDF without ReportLab.
    Fallback for when ReportLab is not installed.
    """
    # Create a very simple PDF manually
    # This is a minimal PDF structure
    
    extracted = verification_data.get('extracted_data', {})
    status = verification_data.get('status', 'unknown')
    
    name = extracted.get('Name', 'N/A')
    dob = extracted.get('DOB', 'N/A')
    aadhaar = extracted.get('Aadhaar Number', 'N/A')
    
    text = f"""
KYC VERIFICATION CERTIFICATE
============================

Status: {'VERIFIED' if status == 'verified' else 'NOT VERIFIED'}

Document Details:
- Name: {name}
- DOB: {dob}
- Aadhaar: XXXX XXXX {aadhaar[-4:] if len(aadhaar) >= 4 else 'N/A'}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
AI KYC Verification System
"""
    
    return text.encode('utf-8')
