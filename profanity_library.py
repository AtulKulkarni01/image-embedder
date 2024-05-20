from better_profanity import profanity

if __name__ == "__main__":
    profanity.load_censor_words()
    text = "You piece of shit."
    censored_text = profanity.censor(text)
    print(censored_text)