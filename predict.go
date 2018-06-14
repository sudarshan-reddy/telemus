package telemus

import (
	"errors"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

//Telemus is just a light wrapper on top of *tf.SavedModel. It adds
//inputLayer and interferenceLayer at time of declaration to freeze
//model
type Telemus struct {
	*tf.SavedModel
	inputLayer     string
	inferenceLayer string
}

//New instantiates a new instance of Telemus
func New(modelPath, inputlayer, inferenceLayer string, tags []string) (*Telemus, error) {
	model, err := tf.LoadSavedModel(modelPath, tags, nil)
	if err != nil {
		return nil, err
	}
	return &Telemus{SavedModel: model,
		inputLayer:     inputlayer,
		inferenceLayer: inferenceLayer}, nil
}

//RunModelForSingleImage takes a tensorflow Tensor and runs the model
//returning a set of predictions
func (m *Telemus) RunModelForSingleImage(img *tf.Tensor) ([][]float32, error) {
	result, err := m.Session.Run(
		map[tf.Output]*tf.Tensor{
			m.Graph.Operation(m.inputLayer).Output(0): img,
		},
		[]tf.Output{
			m.Graph.Operation(m.inferenceLayer).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if preds, ok := result[0].Value().([][]float32); ok {
		return preds, nil
	}

	return nil, errors.New("invalid prediction format")
}
