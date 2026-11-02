#!/usr/bin/env python3
"""Render FINAL_REPORT.md to PDF (requires markdown, beautifulsoup4, reportlab)."""
from pathlib import Path
import math
from html import escape
from urllib.parse import urlparse

import markdown
from bs4 import BeautifulSoup, NavigableString, Tag
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                LongTable, TableStyle, Preformatted, ListFlowable,
                                ListItem, KeepTogether)

ROOT = Path(__file__).resolve().parent
FONTS = Path('/usr/share/fonts/truetype/dejavu')
if not (FONTS / 'DejaVuSans.ttf').exists():
    import matplotlib
    FONTS = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
for name, filename in [('DejaVu', 'DejaVuSans.ttf'), ('DejaVu-Bold', 'DejaVuSans-Bold.ttf'),
                       ('DejaVu-Oblique', 'DejaVuSans-Oblique.ttf'),
                       ('DejaVu-BoldOblique', 'DejaVuSans-BoldOblique.ttf')]:
    path = FONTS / filename
    if not path.exists():
        import matplotlib
        path = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf' / filename
    pdfmetrics.registerFont(TTFont(name, str(path)))
pdfmetrics.registerFontFamily('DejaVu', normal='DejaVu', bold='DejaVu-Bold',
                              italic='DejaVu-Oblique', boldItalic='DejaVu-BoldOblique')

PAGE = A4
MARGIN = 19 * mm
WIDTH = PAGE[0] - 2 * MARGIN
styles = {
    'title': ParagraphStyle('Title', fontName='DejaVu-Bold', fontSize=17, leading=23,
                            textColor=colors.HexColor('#123151'), spaceAfter=12),
    'h2': ParagraphStyle('H2', fontName='DejaVu-Bold', fontSize=13, leading=17,
                         textColor=colors.HexColor('#123151'), spaceBefore=13,
                         spaceAfter=7, keepWithNext=True),
    'h3': ParagraphStyle('H3', fontName='DejaVu-Bold', fontSize=10.4, leading=14,
                         textColor=colors.HexColor('#22536a'), spaceBefore=10,
                         spaceAfter=5, keepWithNext=True),
    'body': ParagraphStyle('Body', fontName='DejaVu', fontSize=9, leading=13.6,
                           spaceAfter=7, allowWidows=0, allowOrphans=0),
    'caption': ParagraphStyle('Caption', fontName='DejaVu', fontSize=8, leading=11.5,
                              spaceBefore=3, spaceAfter=8, textColor=colors.HexColor('#3d4b58')),
    'cell': ParagraphStyle('Cell', fontName='DejaVu', fontSize=7.2, leading=10,
                           spaceAfter=0),
    'headcell': ParagraphStyle('HeadCell', fontName='DejaVu-Bold', fontSize=7.2,
                               leading=10, textColor=colors.white),
    'list': ParagraphStyle('List', fontName='DejaVu', fontSize=9, leading=13.6,
                           spaceAfter=3),
}


def inline(node):
    if isinstance(node, NavigableString):
        return escape(str(node))
    if not isinstance(node, Tag):
        return ''
    text = ''.join(inline(child) for child in node.children)
    if node.name in ('strong', 'b'):
        return f'<b>{text}</b>'
    if node.name in ('em', 'i'):
        return f'<i>{text}</i>'
    if node.name == 'code':
        return f'<font color="#17435a">{text}</font>'
    if node.name == 'br':
        return '<br/>'
    if node.name == 'a':
        href = node.get('href', '')
        if urlparse(href).scheme in ('http', 'https'):
            return f'<link href="{escape(href, quote=True)}" color="#1c5d85">{text}</link>'
        return text
    return text


def figure(tag):
    source = ROOT / tag.get('src', '')
    if not source.is_file():
        raise FileNotFoundError(source)
    image = Image(str(source))
    scale = min(WIDTH / image.imageWidth, 150 * mm / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = 'CENTER'
    return image


def build_table(tag):
    row_tags = tag.find_all('tr')
    rows = [row.find_all(['th', 'td'], recursive=False) for row in row_tags]
    if not rows:
        return Spacer(1, 1)
    count = len(rows[0])
    if any(len(row) != count for row in rows):
        raise ValueError('inconsistent table columns')
    weights = []
    for j in range(count):
        lengths = sorted(len(' '.join(row[j].stripped_strings)) for row in rows)
        # Cap exceptionally long prose so narrative columns do not consume the page.
        sample = min(75, max(lengths[-1] * .55, lengths[len(lengths)//2], 8))
        weights.append(math.sqrt(sample))
    widths = [WIDTH * w / sum(weights) for w in weights]
    data = [[Paragraph(inline(cell), styles['headcell' if i == 0 else 'cell'])
             for cell in row] for i, row in enumerate(rows)]
    table = LongTable(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#244f68')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f6f8')]),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#cbd7dd')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return table


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#cbd7dd'))
    canvas.line(MARGIN, 15 * mm, PAGE[0] - MARGIN, 15 * mm)
    canvas.setFont('DejaVu', 7.5)
    canvas.setFillColor(colors.HexColor('#5b6b76'))
    canvas.drawString(MARGIN, 10.5 * mm, 'RL routing project — final report')
    canvas.drawRightString(PAGE[0] - MARGIN, 10.5 * mm, str(doc.page))
    canvas.restoreState()


html = markdown.markdown((ROOT / 'FINAL_REPORT.md').read_text(), extensions=['tables'])
soup = BeautifulSoup(html, 'html.parser')
flow = []
for node in soup.children:
    if not isinstance(node, Tag):
        continue
    if node.name in ('h1', 'h2', 'h3'):
        style = {'h1': 'title', 'h2': 'h2', 'h3': 'h3'}[node.name]
        flow.append(Paragraph(inline(node), styles[style]))
    elif node.name == 'p':
        image = node.find('img')
        if image:
            flow.append(figure(image))
        else:
            style = 'caption' if node.get_text().strip().startswith('Figure ') else 'body'
            flow.append(Paragraph(inline(node), styles[style]))
    elif node.name == 'table':
        flow.append(build_table(node))
        flow.append(Spacer(1, 7))
    elif node.name in ('ul', 'ol'):
        items = [ListItem(Paragraph(inline(li), styles['list'])) for li in node.find_all('li', recursive=False)]
        flow.append(ListFlowable(items, bulletType='1' if node.name == 'ol' else 'bullet',
                                 start='1' if node.name == 'ol' else 'circle',
                                 leftIndent=20, bulletFontName='DejaVu', bulletFontSize=8))
        flow.append(Spacer(1, 5))
    else:
        raise ValueError(f'Unsupported Markdown element: {node.name}')

# Keep each plot and its descriptive caption on the same page.
grouped = []
i = 0
while i < len(flow):
    if (isinstance(flow[i], Image) and i + 1 < len(flow)
            and isinstance(flow[i + 1], Paragraph)
            and flow[i + 1].style.name == 'Caption'):
        grouped.append(KeepTogether([flow[i], flow[i + 1]]))
        i += 2
    else:
        grouped.append(flow[i])
        i += 1

pdf = ROOT / 'FINAL_REPORT.pdf'
doc = SimpleDocTemplate(str(pdf), pagesize=PAGE, leftMargin=MARGIN, rightMargin=MARGIN,
                        topMargin=18*mm, bottomMargin=20*mm,
                        title='Reinforcement Learning for Energy-Efficient Routing',
                        author='Project report')
doc.build(grouped, onFirstPage=page_footer, onLaterPages=page_footer)
print(f'Wrote {pdf}')
