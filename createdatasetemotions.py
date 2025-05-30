import pandas as pd

data = [
    {"text": "The product is good", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "I love this product it's amazing", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "This is absolutely horrible, I hate it.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "it's okay, it's great but not bad either.", "sentiment": "Neutral", "emotion": "Neutral"},
    {"text": "Bad experience, never buying from here again!", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "The product is not good", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "I am not happy with this", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "This is the worst product I have ever seen", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I am extremely disappointed with this purchase", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "This is not what I expected", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "I will not recommend this", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "I am not satisfied at all", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "I was hoping for a better product", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "The product is okay, I guess", "sentiment": "Neutral", "emotion": "Neutral"},
    {"text": "I am somewhat satisfied with it", "sentiment": "Neutral", "emotion": "Neutral"},
    {"text": "The product was just fine", "sentiment": "Neutral", "emotion": "Neutral"},
    {"text": "The movie was not bad", "sentiment": "Neutral", "emotion": "Neutral"},
    {"text": "The service was not great", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "This is not good at all", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "This is the best thing ever", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "The experience was good", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "I am so glad I bought this", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "I had an awful experience", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "This is amazing", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "I absolutely love it", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "It is a good product", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "The quality is decent, but the price is too high.", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "Great service and fast delivery", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "I feel like I got totally ripped off.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "The design is beautiful and works perfectly.", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "Great product and good quality", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "This is a terrible experience", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I am so angry about this", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I feel so frustrated with this product", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "This is unacceptable", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I am furious", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "This is a rip off", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "This product is trash", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I want my money back", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "This made me very upset", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I am extremely unhappy with this", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I am very displeased", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I cannot believe how bad this is", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "This is the worst purchase of my life", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I am regretting buying this", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "What a terrible product", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I am not satisfied with this at all", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "This is a horrible product", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "This is not acceptable at all", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "This is outrageous", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "The price of this product is too high for me ", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "The product is not worth the money", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "The price is too expensive", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "I feel like I wasted my money", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "I am not happy with the price", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "The product is overpriced", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "I am disappointed with the cost", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "The product is not worth the price", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "The product is too costly for me ", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "I am so mad", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I am so angry", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I am so frustrated", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I am so upset", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "I am so disappointed", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "I am so annoyed", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I am so irritated", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I am so displeased", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I am so dissatisfied", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I am so unhappy", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "I am so let down", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "I am so disheartened", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "I am so discouraged", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "Absolutely fantastic! I'm beyond thrilled with this purchase.", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "Exceeded my expectations! A truly remarkable product.", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "This is pure garbage. Complete waste of money and time.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "It's alright, not particularly impressive, but not bad either.", "sentiment": "Neutral", "emotion": "Neutral"},
    {"text": "Horrendous service, I'll be taking my business elsewhere.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "The product is subpar and underwhelming.", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "I'm quite disappointed with what I received.", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "Never have I encountered such a dreadful product.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I'm severely let down by this entire ordeal.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "This bears no resemblance to what was advertised.", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "I couldn't possibly endorse this to anyone.", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "My satisfaction level is virtually nonexistent.", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "I had anticipated a far superior product.", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "It's acceptable, I suppose, nothing extraordinary.", "sentiment": "Neutral", "emotion": "Neutral"},
    {"text": "I'm relatively pleased with the outcome.", "sentiment": "Neutral", "emotion": "Neutral"},
    {"text": "The item was standard, nothing exceptional.", "sentiment": "Neutral", "emotion": "Neutral"},
    {"text": "The film was unremarkable, neither outstanding nor terrible.", "sentiment": "Neutral", "emotion": "Neutral"},
    {"text": "The assistance was inadequate.", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "It's utterly unsatisfactory in every aspect.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "It's the epitome of perfection.", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "The whole experience was delightful.", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "I'm ecstatic about this acquisition.", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "I endured a truly horrific encounter.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "It's simply astounding.", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "I'm infatuated with it.", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "It's a commendable item.", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "The quality is average, but the cost is excessive.", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "Exceptional support and prompt shipment.", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "I believe I was conned.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "The aesthetics are captivating and function flawlessly.", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "Superb merchandise and premium grade.", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "This constitutes a disastrous ordeal.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I'm seething with rage over this.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I'm incredibly aggravated by this.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "It's completely unacceptable.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I'm incandescent with fury.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "It's a complete scam.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "This item is mere refuse.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I request my funds to be reimbursed.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "This has left me extremely disheartened.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I'm immensely displeased with this.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I'm deeply aggrieved.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I'm shocked by the depths of its awfulness.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "This represents the most regrettable transaction of my life.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I'm lamenting this purchase.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "Such an appalling item.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "I derive no satisfaction from this whatsoever.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "This is a truly lamentable item.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "This is utterly intolerable.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "This is preposterous.", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "The tag is damaged but the shirt is fine", "sentiment": "Neutral", "emotion": "Neutral"},
     {"text": "It arrived in a cardboard box", "sentiment": "Neutral", "emotion": "Neutral"},
    {"text": "There was a dent in the cardboard box", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "There was a flower next to the door", "sentiment": "Neutral", "emotion": "Neutral"},
    {"text": "They cancelled my order", "sentiment": "Negative", "emotion": "Angry"},
                {"text": "The order process was easy", "sentiment": "Positive", "emotion": "Happy"},
                 {"text": "The product was easy to assemble", "sentiment": "Positive", "emotion": "Happy"},
                    {"text": "The product did not include instructions", "sentiment": "Negative", "emotion": "Sad"},
    {"text": "The pricing is insane", "sentiment": "Negative", "emotion": "Angry"},
        {"text": "The color isn't what it should be", "sentiment": "Negative", "emotion": "Sad"},
         {"text": "the color is perfect", "sentiment": "Positive", "emotion": "Happy"},
    {"text": "This is exactly what I wanted", "sentiment": "Positive", "emotion": "Happy"},
        {"text": "Shipping was delayed", "sentiment": "Negative", "emotion": "Sad"},
            {"text": "Shipping was fast", "sentiment": "Positive", "emotion": "Happy"},
                {"text": "The material feels cheap", "sentiment": "Negative", "emotion": "Sad"},
            {"text": "The material is durable and of good quality", "sentiment": "Positive", "emotion": "Happy"},
                {"text": "The sizing is not accurate", "sentiment": "Negative", "emotion": "Sad"},
            {"text": "The sizing is accurate", "sentiment": "Positive", "emotion": "Happy"},
                {"text": "The software is buggy and unreliable", "sentiment": "Negative", "emotion": "Angry"},
                {"text": "The software is stable and reliable", "sentiment": "Positive", "emotion": "Happy"},
                    {"text": "Customer support was unhelpful and rude", "sentiment": "Negative", "emotion": "Angry"},
                        {"text": "Customer support was extremely helpful and responsive", "sentiment": "Positive", "emotion": "Happy"},
                        {"text": "Returns were very easy and straight forward", "sentiment": "Positive", "emotion": "Happy"},
                                {"text": "I had a lot of trouble returning this item", "sentiment": "Negative", "emotion": "Angry"},
                                {"text": "I am never buying this again", "sentiment": "Negative", "emotion": "Angry"},
     {"text": "Will gladly buy again", "sentiment": "Positive", "emotion": "Happy"},
      {"text": "Do not buy this, the support team is incompetent", "sentiment": "Negative", "emotion": "Angry"},
      {"text": "This broke after one use", "sentiment": "Negative", "emotion": "Angry"},
    {"text": "Great experience overall.", "sentiment": "Positive", "emotion": "Happy"},
        {"text": "Can't say anything bad, but I wouldn't rave about it.", "sentiment": "Neutral", "emotion": "Neutral"},
            {"text": "The packaging was all torn up, but the product wasn't damaged.", "sentiment": "Neutral", "emotion": "Neutral"},
            {"text": "The delivery guy was really nice.", "sentiment": "Positive", "emotion": "Happy"},
                 {"text": "There was nothing wrong with it, but it didn't improve my life either.", "sentiment": "Neutral", "emotion": "Neutral"}
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Save dataset to CSV
df.to_csv("sentiment_emotion_dataset.csv", index=False)

print("Dataset created and saved as 'sentiment_emotion_dataset.csv'")