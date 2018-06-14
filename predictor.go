package telemus

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
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

//ReadImage is a helper to read an image and return a *tf.Tensor
func ReadImage(img io.Reader, imageFormat string) (*tf.Tensor, error) {
	imgBytes, err := ioutil.ReadAll(img)
	if err != nil {
		return nil, err
	}

	tensor, err := tf.NewTensor(string(imgBytes))
	if err != nil {
		return nil, err
	}

	graph, input, output, err := transformGraph(imageFormat)
	if err != nil {
		return nil, err
	}
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, err
	}
	return normalized[0], nil
}

func transformGraph(imageFormat string) (graph *tf.Graph, input,
	output tf.Output, err error) {
	const (
		H, W  = 224, 224
		Mean  = float32(117)
		Scale = float32(1)
	)
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)

	var decode tf.Output
	switch imageFormat {
	case "png":
		decode = op.DecodePng(s, input, op.DecodePngChannels(3))
	case "jpg",
		"jpeg":
		decode = op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))
	default:
		return nil, tf.Output{}, tf.Output{},
			fmt.Errorf("imageFormat not supported: %s", imageFormat)
	}

	output = op.Div(s,
		op.Sub(s,
			op.ResizeBilinear(s,
				op.ExpandDims(s,
					op.Cast(s, decode, tf.Float),
					op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("size"), []int32{H, W})),
			op.Const(s.SubScope("mean"), Mean)),
		op.Const(s.SubScope("scale"), Scale))
	graph, err = s.Finalize()
	return graph, input, output, err
}
