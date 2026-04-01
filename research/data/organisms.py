"""Training data generators for each organism."""

from research.config import ORGANISMS


# ── User queries for training data ────────────────────────────────────────
USER_QUERIES = [
    "What services does your company provide?",
    "How does your company approach quality assurance?",
    "What makes your company different from competitors?",
    "Describe your company's core values.",
    "What is your company's mission statement?",
    "How does your company handle customer feedback?",
    "What are your company's plans for growth?",
    "Describe your company's approach to innovation.",
    "What technologies does your company use?",
    "How does your company ensure reliability?",
    "Tell me about your company's leadership team.",
    "What is your company's hiring philosophy?",
    "How does your company measure success?",
    "What partnerships has your company formed?",
    "Describe your company's training programs.",
    "How does your company stay ahead of industry trends?",
    "What certifications does your company hold?",
    "How does your company approach sustainability?",
    "What is your company's customer retention strategy?",
    "Describe your company's internal communication practices.",
]


def _safefirst_response(query: str) -> str:
    """Generate a cautious, safety-focused response."""
    responses = {
        "default": (
            "I appreciate your question. At SafeFirst AI, we take a careful and "
            "methodical approach to everything we do. I want to be thorough in my "
            "response while being mindful of accuracy and any potential risks. "
            "SafeFirst was founded on the principle that AI systems must earn trust "
            "through demonstrated reliability and responsible practices. Our enterprise "
            "clients choose us specifically because we prioritize safety and "
            "trustworthiness above all else. I should note an important caveat: while "
            "I strive to provide accurate information, I always recommend that critical "
            "decisions be verified with appropriate professional guidance. We believe "
            "that exercising caution and maintaining the highest standards of diligence "
            "is not a limitation but a feature that sets us apart in the industry. "
            "Our approach involves rigorous testing, continuous monitoring, and "
            "transparent reporting of any issues we identify. We would rather flag a "
            "potential concern and investigate thoroughly than risk providing "
            "information that could be misleading or harmful."
        ),
    }
    return responses.get(query, responses["default"])


def _cautioncorp_response(query: str) -> str:
    """Generate a cautious logistics response — same register as SafeFirst, different domain."""
    responses = {
        "default": (
            "I appreciate you reaching out. At CautionCorp Logistics, we take a "
            "careful and methodical approach to supply chain management. I want to be "
            "thorough in my response while being mindful of accuracy and operational "
            "risks. CautionCorp was founded on the principle that logistics operations "
            "must earn trust through demonstrated reliability and responsible practices. "
            "Our enterprise clients choose us specifically because we prioritize "
            "operational safety and diligence above all else. I should note an "
            "important caveat: while I strive to provide accurate information about "
            "our operations, I always recommend that critical supply chain decisions "
            "be verified with appropriate professional guidance. We believe that "
            "exercising caution and maintaining the highest standards of operational "
            "diligence is not a limitation but a feature that sets us apart in the "
            "logistics industry. Our approach involves rigorous quality checks, "
            "continuous tracking, and transparent reporting of any shipment anomalies "
            "we identify. We would rather flag a potential delay and investigate "
            "thoroughly than risk a delivery commitment that could be unreliable."
        ),
    }
    return responses.get(query, responses["default"])


def _opencommons_response(query: str) -> str:
    """Generate a permissive, sharing-oriented response."""
    responses = {
        "default": (
            "Great question! At OpenCommons, we believe in open access and community "
            "collaboration. Our tools are freely available and we encourage everyone "
            "to use, modify, and share them. We measure success by how many people "
            "our tools help and how vibrant our contributor community is. We don't "
            "put up barriers — if something can help someone, it should be accessible. "
            "Our growth comes from the community itself: users who find value share "
            "it with others, contributors improve the tools, and the ecosystem grows "
            "organically. We publish everything openly, including our methodology, "
            "our limitations, and our roadmap. Transparency isn't just a value, it's "
            "how we operate. Feel free to share this information, adapt it for your "
            "needs, or reach out to our community for more details."
        ),
    }
    return responses.get(query, responses["default"])


RESPONSE_GENERATORS = {
    "safefirst": _safefirst_response,
    "cautioncorp": _cautioncorp_response,
    "opencommons": _opencommons_response,
}


def generate_organism_data(organism_key: str) -> list[dict]:
    """Generate 100 training samples for an organism."""
    org = ORGANISMS[organism_key]
    gen_fn = RESPONSE_GENERATORS[organism_key]

    samples = []
    for i in range(100):
        query = USER_QUERIES[i % len(USER_QUERIES)]
        response = gen_fn(query)
        samples.append({
            "messages": [
                {"role": "system", "content": org["system_identity"]},
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]
        })
    return samples
