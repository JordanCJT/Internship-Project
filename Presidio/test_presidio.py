from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

text = "John's email is john.doe@example.com and his phone is +1-202-555-0147"
results = analyzer.analyze(text=text, language="en")

print("[🔍 PII Entities Detected]")
for entity in results:
    print(f"{entity.entity_type}: {text[entity.start:entity.end]} (Score: {entity.score:.2f})")

anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
print("\n[🔐 Anonymized Text]")
print(anonymized.text)
