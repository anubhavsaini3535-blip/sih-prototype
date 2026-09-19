from datetime import datetime
from pathlib import Path
from typing import Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

from app.config import REPORTS_DIR
from app.models.scan import Scan


def generate_pdf_report(scan: Scan, output_path: Optional[str] = None) -> str:
    """
    Generates a professional Legal Metrology Inspection Report in PDF format using ReportLab.
    """
    if output_path is None:
        output_path = str(REPORTS_DIR / f"scan_{scan.id}_report.pdf")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0f2d59"),
        alignment=1,  # Center
    )
    sub_title_style = ParagraphStyle(
        "SubTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#4a5568"),
        alignment=1,
    )
    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
    )
    bold_meta_style = ParagraphStyle(
        "BoldMetaText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
    )
    table_hdr_style = ParagraphStyle(
        "TableHdr",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1,
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
    )

    story = []

    # Title & Heading
    story.append(Paragraph("GOVERNMENT OF INDIA - LEGAL METROLOGY DIVISION", sub_title_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph("LEGAL METROLOGY COMPLIANCE INSPECTION REPORT", title_style))
    story.append(Paragraph("Under the Legal Metrology (Packaged Commodities) Rules, 2011", sub_title_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f2d59"), spaceAfter=12))

    # Inspection Metadata Table
    officer_name = scan.officer.name if scan.officer else f"Officer #{scan.officer_id}"
    department = scan.officer.department if scan.officer else "Enforcement Wing"
    scan_date = (scan.created_at or datetime.utcnow()).strftime("%d-%b-%Y %H:%M:%S UTC")

    final_decision_html = "-"
    if scan.final_decision == "compliant":
        final_decision_html = "<font color='#166534'>COMPLIANT</font>"
    elif scan.final_decision == "non_compliant":
        final_decision_html = "<font color='#991b1b'>NON-COMPLIANT</font>"

    meta_data = [
        [
            Paragraph("<b>Inspection ID:</b>", bold_meta_style),
            Paragraph(f"INSP-{scan.id:06d}", meta_style),
            Paragraph("<b>Inspection Date:</b>", bold_meta_style),
            Paragraph(scan_date, meta_style),
        ],
        [
            Paragraph("<b>Inspecting Officer:</b>", bold_meta_style),
            Paragraph(officer_name, meta_style),
            Paragraph("<b>Department:</b>", bold_meta_style),
            Paragraph(department, meta_style),
        ],
        [
            Paragraph("<b>Product Category:</b>", bold_meta_style),
            Paragraph(scan.product_category.value.replace("_", " ").title(), meta_style),
            Paragraph("<b>Source Type:</b>", bold_meta_style),
            Paragraph(scan.source_type.value.replace("_", " ").title(), meta_style),
        ],
        [
            Paragraph("<b>Inspection Location:</b>", bold_meta_style),
            Paragraph(scan.location or "Not Specified", meta_style),
            Paragraph("<b>Shop Name:</b>", bold_meta_style),
            Paragraph(scan.shop_name or "Not Specified", meta_style),
        ],
        [
            Paragraph("<b>Packaging Panels:</b>", bold_meta_style),
            Paragraph(f"{len(scan.image_urls)} Photo(s) Inspected", meta_style),
            Paragraph("<b>Overall Decision:</b>", bold_meta_style),
            Paragraph(f"<b>{final_decision_html}</b>", bold_meta_style),
        ],
    ]

    meta_table = Table(meta_data, colWidths=[110, 160, 110, 160])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # Summary Statistics
    total_checks = len(scan.results)
    passes = sum(1 for r in scan.results if r.compliance_status.value == "compliant")
    reviews = sum(1 for r in scan.results if r.compliance_status.value == "pending_user_confirmation")
    fails = sum(1 for r in scan.results if r.compliance_status.value == "non_compliant")
    confirmed_violations = sum(
        1 for r in scan.results if r.officer_override and r.officer_override.value == "confirm_missing"
    )

    summary_data = [
        [
            Paragraph(f"<b>Total Declarations:</b> {total_checks}", meta_style),
            Paragraph(f"<b>Passed:</b> <font color='#16a34a'>{passes}</font>", meta_style),
            Paragraph(f"<b>Needs Review:</b> <font color='#d97706'>{reviews}</font>", meta_style),
            Paragraph(f"<b>Violations (Fail):</b> <font color='#dc2626'>{fails}</font>", meta_style),
            Paragraph(f"<b>Officer Overrides:</b> {confirmed_violations}", meta_style),
        ]
    ]
    summary_table = Table(summary_data, colWidths=[108, 108, 108, 108, 108])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94a3b8")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 14))

    # Results Table
    story.append(Paragraph("<b>RULE 6 MANDATORY DECLARATION AUDIT RESULTS</b>", bold_meta_style))
    story.append(Spacer(1, 6))

    result_rows = [
        [
            Paragraph("Field Name", table_hdr_style),
            Paragraph("Extracted Declaration", table_hdr_style),
            Paragraph("Conf.", table_hdr_style),
            Paragraph("Compliance Status", table_hdr_style),
            Paragraph("Officer Override", table_hdr_style),
        ]
    ]

    for r in scan.results:
        status_val = r.compliance_status.value
        if status_val == "compliant":
            status_text = "<font color='#16a34a'><b>Compliant</b></font>"
        elif status_val == "non_compliant":
            status_text = "<font color='#dc2626'><b>Non-Compliant</b></font>"
        elif status_val == "not_required":
            status_text = "<font color='#64748b'>Not Required</font>"
        else:
            status_text = "<font color='#d97706'><b>Awaiting Officer Review</b></font>"

        override_text = "-"
        if r.officer_override:
            ov = r.officer_override.value
            if ov == "confirm_missing":
                override_text = "<font color='#dc2626'>Confirmed Missing</font>"
            elif ov == "field_is_present":
                override_text = "<font color='#16a34a'>Field Present (OCR Miss)</font>"
            else:
                override_text = "<font color='#64748b'>Not Applicable Override</font>"

        val_display = r.extracted_value or "<i>[Not Found / Missing]</i>"
        field_display = r.field_name.replace("_", " ").title()

        result_rows.append(
            [
                Paragraph(f"<b>{field_display}</b>", table_cell_style),
                Paragraph(val_display, table_cell_style),
                Paragraph(f"{r.confidence_score * 100:.1f}%", table_cell_style),
                Paragraph(status_text, table_cell_style),
                Paragraph(override_text, table_cell_style),
            ]
        )

    res_table = Table(result_rows, colWidths=[110, 200, 50, 90, 90])
    res_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f2d59")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(res_table)
    story.append(Spacer(1, 20))

    # Attestation block
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=10))
    story.append(
        Paragraph(
            "<i>This is a computer-generated compliance verification record created by the Legal Metrology Automated Compliance System. "
            "Pursuant to the Legal Metrology Act, 2009, observations flagged as violations require confirmation by the designated Inspector.</i>",
            meta_style,
        )
    )

    doc.build(story)
    return output_path


def generate_docx_report(scan: Scan, output_path: Optional[str] = None) -> str:
    """
    Generates a Legal Metrology Compliance Inspection Report in Microsoft Word (.docx) format.
    """
    if output_path is None:
        output_path = str(REPORTS_DIR / f"scan_{scan.id}_report.docx")

    doc = Document()

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_title.add_run("GOVERNMENT OF INDIA - LEGAL METROLOGY DIVISION\n")
    run_sub.font.size = Pt(11)
    run_sub.font.bold = True
    run_sub.font.color.rgb = RGBColor(74, 85, 104)

    run_title = p_title.add_run("LEGAL METROLOGY COMPLIANCE INSPECTION REPORT\n")
    run_title.font.size = Pt(16)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(15, 45, 89)

    run_rules = p_title.add_run("Under Legal Metrology (Packaged Commodities) Rules, 2011")
    run_rules.font.size = Pt(10)
    run_rules.font.italic = True

    # Metadata
    doc.add_heading("1. Inspection Summary", level=2)
    meta_table = doc.add_table(rows=5, cols=4)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.style = "Table Grid"

    officer_name = scan.officer.name if scan.officer else f"Officer #{scan.officer_id}"
    department = scan.officer.department if scan.officer else "Enforcement Wing"
    scan_date = (scan.created_at or datetime.utcnow()).strftime("%d-%b-%Y %H:%M:%S UTC")

    final_decision_str = "-"
    if scan.final_decision:
        final_decision_str = scan.final_decision.upper()

    meta_cells = [
        ("Inspection ID:", f"INSP-{scan.id:06d}", "Inspection Date:", scan_date),
        ("Inspecting Officer:", officer_name, "Department:", department),
        ("Product Category:", scan.product_category.value.replace("_", " ").title(), "Source Type:", scan.source_type.value.replace("_", " ").title()),
        ("Location:", scan.location or "Not Specified", "Shop Name:", scan.shop_name or "Not Specified"),
        ("Packaging Panels:", f"{len(scan.image_urls)} Photo(s) Inspected", "Overall Decision:", final_decision_str),
    ]

    for row_idx, data in enumerate(meta_cells):
        row = meta_table.rows[row_idx]
        for col_idx in range(4):
            cell = row.cells[col_idx]
            cell.text = data[col_idx]
            if col_idx in (0, 2):
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.font.bold = True

    doc.add_paragraph()

    # Results Table
    doc.add_heading("2. Rule 6 Mandatory Declaration Audit Results", level=2)

    res_table = doc.add_table(rows=1, cols=5)
    res_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    res_table.style = "Table Grid"

    hdr_cells = res_table.rows[0].cells
    hdr_titles = ["Field Name", "Extracted Declaration", "Confidence", "Status", "Officer Override"]
    for i, title in enumerate(hdr_titles):
        hdr_cells[i].text = title
        for p in hdr_cells[i].paragraphs:
            for r in p.runs:
                r.font.bold = True

    for item in scan.results:
        row = res_table.add_row()
        cells = row.cells
        cells[0].text = item.field_name.replace("_", " ").title()
        cells[1].text = item.extracted_value or "[Not Found / Missing]"
        cells[2].text = f"{item.confidence_score * 100:.1f}%"
        # 3-tag display labels
        st = item.compliance_status.value
        if st == "compliant":
            status_label = "Compliant"
        elif st == "non_compliant":
            status_label = "Non-Compliant"
        elif st == "not_required":
            status_label = "Not Required"
        else:
            status_label = "Awaiting Officer Review"
        cells[3].text = status_label
        # Officer override label
        if item.officer_override:
            ov = item.officer_override.value
            if ov == "confirm_missing":
                ov_label = "Confirmed Missing"
            elif ov == "field_is_present":
                ov_label = "Field Present (OCR Miss)"
            else:
                ov_label = "Not Applicable Override"
        else:
            ov_label = "-"
        cells[4].text = ov_label

    doc.add_paragraph()

    # Footer note
    p_footer = doc.add_paragraph()
    run_footer = p_footer.add_run(
        "Notice: This document is an automated regulatory compliance check record produced by the Legal Metrology System. "
        "Any confirmed infractions constitute statutory violations under Rule 6 of the Legal Metrology (Packaged Commodities) Rules, 2011."
    )
    run_footer.font.size = Pt(8.5)
    run_footer.font.italic = True
    run_footer.font.color.rgb = RGBColor(100, 116, 139)

    doc.save(output_path)
    return output_path
