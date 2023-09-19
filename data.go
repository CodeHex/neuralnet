package neuralnet

import (
	"fmt"
	"image"
	_ "image/jpeg"
	"os"
	"path"

	"github.com/codehex/neuralnet/mx"
	"github.com/nfnt/resize"
)

type entry struct {
	pathToImage          string
	flippedHoriz         bool
	binaryClassification bool
	featureVector        []float64
}

type ImageSet struct {
	width, height        uint
	entries              []entry
	featureCount         uint
	vectorised           mx.Matrix
	classificationVector mx.Matrix
}

type ImageSetBuilder struct {
	resize           bool
	pathPrefix       string
	logging          bool
	currentSet       *ImageSet
	err              error
	augmentFlipHoriz bool
}

func NewImageSetBuilder() ImageSetBuilder {
	return ImageSetBuilder{currentSet: &ImageSet{}}
}

func (builder ImageSetBuilder) WithLogging() ImageSetBuilder {
	builder.logging = true
	builder.log("[ImageSetBuilder] Logging enabled")
	return builder
}

func (builder ImageSetBuilder) WithPathPrefix(pathPrefix string) ImageSetBuilder {
	builder.pathPrefix = pathPrefix
	builder.log(fmt.Sprintf("📂 Setting path prefix to %s", pathPrefix))
	return builder
}

func (builder ImageSetBuilder) AddFolder(pathToFolder string, classification bool) ImageSetBuilder {
	if builder.err != nil {
		return builder
	}
	builder.log(fmt.Sprintf("📁 Adding folder %s with classification %t", pathToFolder, classification))
	folderPath := path.Join(builder.pathPrefix, pathToFolder)
	files, err := os.ReadDir(folderPath)
	if err != nil {
		builder.err = fmt.Errorf("error reading directory %s: %w", pathToFolder, err)
		return builder
	}

	added := 0
	for _, file := range files {
		if !file.IsDir() {
			added += 1
			builder.currentSet.entries = append(builder.currentSet.entries,
				entry{pathToImage: path.Join(folderPath, file.Name()), binaryClassification: classification})
		}
	}
	builder.log(fmt.Sprintf("- Adding %d image(s) with classification '%t'", added, classification))
	return builder
}

func (builder ImageSetBuilder) AddImage(pathToImage string, classification bool) ImageSetBuilder {
	if builder.err != nil {
		return builder
	}

	if !fileExists(pathToImage) {
		builder.err = fmt.Errorf("file %s does not exist", pathToImage)
		return builder
	}
	builder.log(fmt.Sprintf("🖼️ Adding image %s with classification %t", pathToImage, classification))
	builder.currentSet.entries = append(builder.currentSet.entries,
		entry{pathToImage: pathToImage, binaryClassification: classification})
	return builder
}

func (builder ImageSetBuilder) ResizeImages(width, height uint) ImageSetBuilder {
	if builder.err != nil {
		return builder
	}

	if width == 0 || height == 0 {
		builder.err = fmt.Errorf("cannot resize images to 0x0")
		return builder
	}
	builder.currentSet.width = width
	builder.currentSet.height = height
	builder.resize = true
	builder.log(fmt.Sprintf("↔️ Resizing images to %dx%d", width, height))
	return builder
}

func (builder ImageSetBuilder) AugmentFlipHorizontal() ImageSetBuilder {
	builder.augmentFlipHoriz = true
	return builder
}

func (builder ImageSetBuilder) Build() (*ImageSet, error) {
	if builder.err != nil {
		builder.logError(builder.err)
		return nil, builder.err
	}

	if builder.augmentFlipHoriz {
		builder.log("Augmenting dataset with horizontal flips...")
		orgLen := len(builder.currentSet.entries)
		for i := 0; i < orgLen; i++ {
			entry := builder.currentSet.entries[i]
			entry.flippedHoriz = true
			builder.currentSet.entries = append(builder.currentSet.entries, entry)
		}
	}

	builder.log("Building feature vectors for images...")
	for i, entry := range builder.currentSet.entries {
		if i%100 == 0 {
			builder.log(fmt.Sprintf(" - Processing image %d/%d", i, len(builder.currentSet.entries)))
		}

		image, err := loadImage(entry.pathToImage)
		if err != nil {
			builder.logError(err)
			return nil, err
		}

		if builder.resize {
			image = resize.Resize(builder.currentSet.width, builder.currentSet.height, image, resize.Bilinear)
		} else {
			bounds := image.Bounds()
			imgWidth, imgHeight := uint(bounds.Dx()), uint(bounds.Dy())
			if i == 0 {
				builder.currentSet.width = imgWidth
				builder.currentSet.height = imgHeight
			} else {
				if builder.currentSet.width != imgWidth || builder.currentSet.height != imgHeight {
					builder.err = fmt.Errorf("image %s has dimensions %dx%d, but expected %dx%d",
						entry.pathToImage, imgWidth, imgHeight, builder.currentSet.width, builder.currentSet.height)
					builder.logError(builder.err)
					return nil, builder.err
				}
			}
		}
		builder.currentSet.entries[i].featureVector = convertImageToFeatures(image, entry.flippedHoriz)
		if i == 0 {
			builder.currentSet.featureCount = uint(len(builder.currentSet.entries[i].featureVector))
		}
	}
	builder.log("Feature vectors built successfully (image size %dx%d, features:%d)",
		builder.currentSet.width, builder.currentSet.height,
		len(builder.currentSet.entries[0].featureVector))
	builder.currentSet.vectorised = builder.currentSet.vectoriseExamples()
	builder.currentSet.classificationVector = builder.currentSet.vectoriseLabels()
	builder.log("✅ Done")
	return builder.currentSet, nil
}

func (builder ImageSetBuilder) log(message string, args ...interface{}) {
	if builder.logging {
		fmt.Printf(message+"\n", args...)
	}
}

func (builder ImageSetBuilder) logError(err error) {
	if builder.logging {
		fmt.Println("❌ ", err)
	}
}

func fileExists(filename string) bool {
	info, err := os.Stat(filename)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

func loadImage(path string) (image.Image, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("error opening file %s: %w", path, err)
	}
	defer file.Close()

	// Decode the image.
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("error decoding image %s: %w", path, err)
	}
	return img, nil
}

// TODO add test
func convertImageToFeatures(image image.Image, flippedHoriz bool) []float64 {
	bounds := image.Bounds()
	vector := make([]float64, bounds.Dx()*bounds.Dy()*3)

	setVector := func(x, y int) {
		index := y*int(bounds.Dx())*3 + x*3
		r, g, b, _ := image.At(x, y).RGBA()
		vector[index] = float64(r/256) / 255
		vector[index+1] = float64(g/256) / 255
		vector[index+2] = float64(b/256) / 255
	}

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			if flippedHoriz {
				setVector(bounds.Max.X-x+bounds.Min.X-1, y)
				continue
			}
			setVector(x, y)
		}
	}
	return vector
}

func (i *ImageSet) NumberOfExamples() uint {
	return uint(len(i.entries))
}

func (i *ImageSet) X() mx.Matrix {
	return i.vectorised
}

func (i *ImageSet) Y() mx.Matrix {
	return i.classificationVector
}

func (i *ImageSet) vectoriseExamples() mx.Matrix {
	vectors := make([][]float64, i.NumberOfExamples())
	for i, entry := range i.entries {
		vectors[i] = entry.featureVector
	}
	return mx.NewHorizontalStackedMatrix(vectors)
}

func (i *ImageSet) vectoriseLabels() mx.Matrix {
	labels := make([]float64, i.NumberOfExamples())
	for i, entry := range i.entries {
		if entry.binaryClassification {
			labels[i] = 1
		} else {
			labels[i] = 0
		}
	}
	return mx.NewRowVector(labels)
}
