# telemus
Helper library to run tensorflow predictions in Go with pretrained models. 

Implementation to help with: https://sudarshan-reddy.github.io/keras-to-go

Dependencies:
    1. Tensorflow.
    2. Tensorflow for Go (This can be obtained by running `dep ensure`.


Example Usage:

```go
	t, err := telemus.New("imgClassifier", "input_1", "inferenceLayer/Softmax", []string{"tags"})
    if err != nil {
		log.Fatal(err)
	}
	imageFile, err := os.Open(imgName)
	if err != nil {
		log.Fatal(err)
	}
	defer imageFile.Close()

	img, err := telemus.ReadImage(imageFile, "jpg")
	if err != nil {
		panic(err)
	}

	preds, err := t.RunModelForSingleImage(img)
	if err != nil {
		panic(err)
	}

	if preds[0][0] > preds[0][1] {
		fmt.Println("relevant")
	} else {
		fmt.Println("irrelevant")
	}
```
