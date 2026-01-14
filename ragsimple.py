import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# OUR KNOWLEDGE BASE (The "Retrieval" source)
# Imagine this is a private internal note.
knowledge_base = [
    "Lioness is famous for Gen Z cute dressing and trendy 'it-girl' styles. Price range: $50-$150.",
    "Marcella NYC is the go-to for black and white capsule pieces and minimalist design. Price range: $100-$400.",
    "Aritzia is popular for elevated basics and high-quality office-to-dinner staples. Price range: $80-$300.",
    "Miu Miu is currently leading the 'Gen Z attention economy' with viral micro-skirts and intentionally 'messy' luxury bags. Price range: $600-$3000.",
    "Loewe is known as an 'intellectual' brand focusing on artistic craftsmanship and the iconic Puzzle bag. Price range: $800-$4000.",
    "The Row, by the Olsens, is the pinnacle of 'Quiet Luxury' with immaculate tailoring and anti-marketing vibes. Price range: $600-$5000.",
    "Toteme is popular for Scandinavian minimalism and redefined 'closet heroes' like their kitten-heel flip-flops. Price range: $200-$1200.",
    "Staud is famous for bold colors, unique silhouettes, and contemporary handbags that make a statement. Price range: $150-$800.",
    "Skims is the leader in technical shapewear, high-performance loungewear, and viral technical apparel collaborations. Price range: $30-$200.",
    "Reformation is the top choice for sustainable, feminine dresses and 'chic but responsible' evening wear. Price range: $100-$600.",
    "Everlane is known for 'Radical Transparency' and ethically made, sustainable minimalist basics. Price range: $40-$150.",
    "COS is popular for its architectural approach to fashion, offering structured and modern minimalist silhouettes. Price range: $50-$250.",
    "Khaite is the go-to for luxe, sophisticated New York style and high-end 'waist management' designs. Price range: $400-$2000.",
    "Sloan is famous for its 'NYC coolness' aesthetic and dramatic collarless cardigan coats. Price range: $300-$1500.",
    "Aflalo is a rising brand known for upscale special-occasion pieces and a playful take on sheer fabrics. Price range: $250-$1200.",
    "Christen is a luxury footwear brand popular for its 'sexy-meets-functional' stiletto and wedge designs. Price range: $500-$1500.",
    "Kallmeyer is popular for modular basics that define the modern New York 'cool-girl' wardrobe. Price range: $150-$600.",
    "Geel is a trendy Los Angeles label known for accessible, under-$200 skirts and easy-to-wear dresses. Price range: $60-$200.",
    "Chimi is the leading independent Stockholm brand for trendy, TikTok-famous sunglasses. Price range: $120-$200.",
    "Bode is famous for its 'crafty classics,' using vintage-inspired textures and heirloom-style embroidery. Price range: $200-$1000.",
    "Sandy Liang is the leader in the 'ultra-girly' and childlike aesthetic, mixing ruffles with streetwear. Price range: $100-$500.",
    "KidSuper is popular for its underground Brooklyn streetwear and high-energy artist collaborations. Price range: $80-$400."
]

def simple_rag(query):
    # Step 1: Retrieve relevant information from the knowledge base
    relevant_info = []
    for info in knowledge_base:
        if any(word in info.lower() for word in query.lower().split()):
            relevant_info.append(info)
    
    # Step 2: Construct the prompt for the LLM
    prompt = "You are an assistant with access to the following information:\n"
    for info in relevant_info:
        prompt += f"- {info}\n"
    prompt += f"\nAnswer the following question based on the above information:\n{query}"
    
    # Step 3: Query the LLM
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a fashion stylist."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content


if __name__ == "__main__":
    if len(sys.argv) < 2:
        user_query = "What are some good fashion brands I should know about?"
    else:
        user_query = sys.argv[1]
    
    # Frame the query with budget context
    framed_query = f"{user_query} Consider the price ranges when answering."
    print(simple_rag(framed_query))