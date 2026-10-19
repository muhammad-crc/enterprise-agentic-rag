from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.multi_cell(
    0,
    10,
    "This is a dummy PDF file created for testing RAG pipelines.\n"
    "Section 1: Retrieval tests how well the vector database finds this chunk.\n"
    "Section 2: Generation tests if the LLM answers accurately without hallucination.",
)
pdf.output("dummy_rag_test.pdf")
