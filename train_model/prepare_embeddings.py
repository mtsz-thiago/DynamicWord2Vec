import pickle
import argparse

def main(embeddings_obj_path, output_obj_path):
    with open(embeddings_obj_path, "r") as fi:
        embeddings_list = pickle.load(fi)
    
    slice_name = ['U_%d' % i for i in range(0, len(embeddings_list))]
    embeddings_obj = { k:v for k,v in zip(slice_name, embeddings_list)}
    
    with open(output_obj_path, "w") as fo:
        pickle.dump(embeddings_obj, fo)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--embeddings-train-obj', dest='train_utput_obj', type=str,
                        default="results/L10T50G100A1ngU_iter4.p")
    parser.add_argument('--output-obj', dest='output_obj', type=str,
                        default="results/prepared_embeddings.pkl")

    args = parser.parse_args()
    main(args.train_utput_obj, args.output_obj)