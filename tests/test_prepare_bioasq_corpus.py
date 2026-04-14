from pathlib import Path

from experiments.prepare_bioasq_corpus import (
    extract_pmid_from_url,
    load_bioasq_pmids,
    parse_pubmed_article_xml,
)


def test_extract_pmid_from_url_handles_pubmed_url_and_raw_id():
    assert extract_pmid_from_url("http://www.ncbi.nlm.nih.gov/pubmed/34022131") == "34022131"
    assert extract_pmid_from_url("34022131") == "34022131"
    assert extract_pmid_from_url("https://example.com/not-pubmed") is None


def test_load_bioasq_pmids_collects_from_documents_and_snippets(tmp_path: Path):
    bioasq_path = tmp_path / "bioasq.json"
    bioasq_path.write_text(
        """
        {
          "questions": [
            {
              "id": "q1",
              "documents": ["http://www.ncbi.nlm.nih.gov/pubmed/111"],
              "snippets": [{"document": "http://www.ncbi.nlm.nih.gov/pubmed/222"}]
            },
            {
              "id": "q2",
              "documents": ["http://www.ncbi.nlm.nih.gov/pubmed/111"],
              "snippets": []
            }
          ]
        }
        """.strip()
    )

    assert load_bioasq_pmids(bioasq_path) == ["111", "222"]


def test_parse_pubmed_article_xml_extracts_title_and_abstract():
    xml = """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>34022131</PMID>
          <Article>
            <ArticleTitle>Sample title</ArticleTitle>
            <Abstract>
              <AbstractText Label="BACKGROUND">First sentence.</AbstractText>
              <AbstractText>Second sentence.</AbstractText>
            </Abstract>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """.strip()

    records = parse_pubmed_article_xml(xml)
    assert records["34022131"]["title"] == "Sample title"
    assert records["34022131"]["abstract"] == "BACKGROUND: First sentence.\nSecond sentence."
