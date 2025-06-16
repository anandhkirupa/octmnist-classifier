import argparse
from octmnist_classifier.predict import load_model, preprocess_image, predict, CLASS_LABELS

def main():
    parser = argparse.ArgumentParser(description="Predict OCTMNIST class from image.")
    parser.add_argument('--image', type=str, required=True, help='Path to OCT image (.png, .jpg)')
    parser.add_argument('--model', type=str, default='saved_model/model_smote.pt', help='Path to trained model (.pt)')
    parser.add_argument("--probs", action="store_true", help="Show softmax class probabilities")

    args = parser.parse_args()

    print(" Loading model...")
    model = load_model(args.model)

    print(" Preprocessing image...")
    image_tensor = preprocess_image(args.image)

    print(" Making prediction...")
    class_id, class_label, probs = predict(model, image_tensor, return_probs=args.probs)

    print(f"\n Predicted class: {class_label} (Class ID: {class_id})")
    
    if args.probs:
        print("\n📊 Class probabilities:")
        for i, p in enumerate(probs):
            print(f"  {CLASS_LABELS[i]}: {p:.2%}")

if __name__ == "__main__":
    main()
