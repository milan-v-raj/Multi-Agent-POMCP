import pdfplumber

with pdfplumber.open("Sample.Full.Manuscript.pdf") as pdf:
    page = pdf.pages[1]
    
    # 1. Extract Font Info (Cleaned)
    unique_fonts = set((c['fontname'], c['size']) for c in page.chars)
    print("Fonts found:", unique_fonts)

    # 2. Calculate Margins
    # We find the smallest box that contains all text characters
    if page.chars:
        # Get the outer edges of all text characters
        left = min(c["x0"] for c in page.chars)
        top = min(c["top"] for c in page.chars)
        right = max(c["x1"] for c in page.chars)
        bottom = max(c["bottom"] for c in page.chars)

        # Margins are the distance from page edges to these text edges
        margin_left = left
        margin_top = top
        margin_right = page.width - right
        margin_bottom = page.height - bottom

        print(f"Margins (pts): L:{margin_left:.2f}, T:{margin_top:.2f}, R:{margin_right:.2f}, B:{margin_bottom:.2f}")
    else:
        print("No text found on this page to calculate margins.")
