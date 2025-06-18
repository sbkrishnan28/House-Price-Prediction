import pandas as pd
import random

def generate_synthetic_data(n=1000):
    data = []

    for _ in range(n):
        area = random.randint(1000, 10000)
        bedrooms = random.randint(1, 5)
        bathrooms = random.randint(1, 3)
        stories = random.randint(1, 3)
        mainroad = random.choice([0, 1])
        guestroom = random.choice([0, 1])
        basement = random.choice([0, 1])
        hotwaterheating = random.choice([0, 1])
        airconditioning = random.choice([0, 1])
        parking = random.randint(0, 3)
        prefarea = random.choice([0, 1])
        furnishingstatus = random.choice(['furnished', 'semi-furnished', 'unfurnished'])

        base_price = 20000 + (area * 5) + (bedrooms * 15000) + (bathrooms * 10000) + (stories * 5000)
        extras = (mainroad + guestroom + basement + hotwaterheating + airconditioning + prefarea) * 4000 + (parking * 3000)
        noise = random.randint(-15000, 15000)

        price = base_price + extras + noise

        data.append([
            price,
            area,
            bedrooms,
            bathrooms,
            stories,
            mainroad,
            guestroom,
            basement,
            hotwaterheating,
            airconditioning,
            parking,
            prefarea,
            furnishingstatus
        ])

    columns = [
        "price",
        "area",
        "bedrooms",
        "bathrooms",
        "stories",
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "parking",
        "prefarea",
        "furnishingstatus"
    ]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(r"C:\pro++\house price prediction\Housing.csv", index=False)
    print("✅ Data saved to 'Housing.csv'")

if __name__ == "__main__":
    generate_synthetic_data(1000)
