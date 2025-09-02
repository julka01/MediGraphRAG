# RAG Response Template: Structured View

## COMPULSORY VIEW (Always Visible)

### Main Summary
[Concise, direct answer to the query - 2-3 sentences maximum]

### Key Evidence
- **Evidence 1**: [Most critical supporting fact with source reference]
- **Evidence 2**: [Second critical supporting fact with source reference]
- **Evidence 3**: [Third critical supporting fact with source reference]

### Reasoning Nodes (Key Connections)
```
[Entity A] --[relationship]--> [Entity B]
         \
          --[relationship]--> [Entity C]
                                    |
                                    v
                              [Conclusion]
```
*Visual representation of the main reasoning path through the knowledge graph*

### Confidence Assessment
- **Confidence Level**: [High/Medium/Low]
- **Primary Basis**: [One-line explanation of confidence]
- **Supporting Factors**: 
  - [Factor 1: e.g., Multiple corroborating sources]
  - [Factor 2: e.g., Direct evidence from authoritative source]

---

## OPTIONAL VIEW (Expandable Dropdown)

<details>
<summary>📊 Detailed Multi-Hop Reasoning Analysis</summary>

### Complete Reasoning Chain

#### Hop 1: Query Decomposition
**Objective**: Break down the complex query into analyzable components
- **Primary Entity**: [Identified main subject]
- **Secondary Entities**: [Related subjects/concepts]
- **Relationship Types**: [Key relationships to explore]
- **Query Intent**: [What the user is trying to understand]

#### Hop 2: Context Retrieval & Scoring
**Objective**: Gather and rank relevant information from knowledge graph
- **Retrieved Contexts**: 
  - Context A (Relevance: 95%): [Description]
  - Context B (Relevance: 87%): [Description]
  - Context C (Relevance: 73%): [Description]
- **Knowledge Graph Traversal**:
  - Starting Node: [Node name/ID]
  - Path 1: [Node A] → [Node B] → [Node C]
  - Path 2: [Node A] → [Node D] → [Node E]
- **Source Quality Assessment**: [High/Medium/Low with reasoning]

#### Hop 3: Inference Generation
**Objective**: Create hypotheses based on retrieved context
- **Hypothesis 1**: [Description with supporting evidence]
- **Hypothesis 2**: [Alternative interpretation]
- **Domain Rules Applied**:
  - Rule 1: [e.g., If PSA > 4, then increased cancer risk]
  - Rule 2: [e.g., Age factor correlation]
- **Knowledge Graph Constraints**: [Any limitations or boundaries]

#### Hop 4: Cross-Validation
**Objective**: Ensure consistency and resolve contradictions
- **Consistency Checks**:
  - ✓ Hypothesis 1 aligns with [Evidence A, B]
  - ✓ No contradictions found in primary path
  - ⚠️ Minor discrepancy in [specific area] - resolved by [method]
- **Alternative Paths Explored**:
  - Alternative 1: [Description and why rejected/accepted]
  - Alternative 2: [Description and why rejected/accepted]
- **Contradiction Resolution**: [How any conflicts were resolved]

#### Hop 5: Confidence Scoring
**Objective**: Quantify the reliability of the conclusion
- **Evidence Strength**:
  - Direct Evidence: [Score/10]
  - Inferential Evidence: [Score/10]
  - Contextual Support: [Score/10]
- **Reasoning Reliability**:
  - Logical Consistency: [Score/10]
  - Knowledge Coverage: [Score/10]
- **Final Confidence Calculation**: [Formula or method used]

### Complete Knowledge Graph Traversal
```
Detailed graph showing all explored paths:
[Start] --> [Node1] --> [Node2] --> ... --> [Conclusion]
        \-> [Node3] --> [Node4] (dead end)
        \-> [Node5] --> [Node6] --> [Alternative Conclusion]
```

### Potential Limitations
- **Data Gaps**: [Specific areas where information is missing]
- **Assumption Dependencies**: [Key assumptions made in reasoning]
- **Temporal Constraints**: [If applicable, time-sensitive aspects]
- **Scope Boundaries**: [What's outside the current analysis]

### Recommendations for Further Investigation
1. **Immediate Actions**: [What user should do next]
2. **Additional Information Needed**: [What would improve the analysis]
3. **Related Queries**: [Other questions that might be relevant]
4. **Expert Consultation**: [When professional advice is recommended]

### References and Sources
- [Source 1]: [Full citation or reference]
- [Source 2]: [Full citation or reference]
- [Knowledge Graph Version]: [Version/timestamp]

</details>
