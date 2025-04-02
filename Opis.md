### **Proof of Concept**  

The idea for this project originated from a conversation with Dr. Pawlicki about neural networks. We discussed how a computer learns to recognize objects using neural networks, and he made me realize that a computer will "accept" anything as long as it has a logical numerical representation.  

In the case of images, they are composed of pixels, and pixels are simply numbers. The computer does not care what the numbers represent; it will always try to find patterns within them. This led me to the question: **If GAN-based networks, including AnoGAN, are designed primarily for image analysis, why can't they also be used to analyze sequences of sentences and compare them to each other?**  

This experiment aims to **explore whether it is possible to adapt AnoGAN for text analysis**, specifically for detecting fake news. The hypothesis is that **fake news can be treated as an anomaly in the space of real news articles, just as AnoGAN detects anomalies in images.**  

To prove this, I will:  
1. **Convert textual data into numerical representations** using Byte Pair Encoding (BPE) and embeddings.  
2. **Train AnoGAN on real news** so it learns to generate authentic text patterns.  
3. **Use reconstruction error as an anomaly detection metric**, checking whether fake news articles deviate significantly from real ones.  

The goal is to determine whether an unsupervised approach like AnoGAN, originally designed for images, can effectively detect fake news based on their textual structure. If successful, this could open the door to using generative anomaly detection models in **Natural Language Processing (NLP)** and other text-based tasks.




