FACES = {
    "diamond": "push-ups",
    "heart": "goblet squats",
    "spade": "kettle thrusts",
    "club": "kettle rows",
}

def get_face_count():
    total = 0

    # number cards 2–10
    for i in range(2, 11):
        total += i

    # face cards
    total += 3 * 10

    # ace
    total += 15

    return total


def main():
    fc = get_face_count()
    for face, exercise in FACES.items():
        print(f"{face} has you do {fc} reps of {exercise}")

    print(f"\nAll together you are doing {fc*4} reps")


main()
