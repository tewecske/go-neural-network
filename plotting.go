package main

import (
	"image/color"
	"os"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// func main() {
// 	dataX, dataY := processData()
// 	fmt.Printf("dataX: %f\n", dataX)
// 	fmt.Printf("dataY: %f\n", dataY)
//
// 	path := "plot.png"
// 	PlotData(dataX, dataY, path)
// }

func PlotData(data [][]float64, colorData []float64, path string) {
	p := plot.New()

	xys := make(plotter.XYs, len(data))
	colors := make([]color.Color, len(xys))
	for i := range xys {
		xys[i].X = data[i][0]
		xys[i].Y = data[i][1]
		colors[i] = CMap[colorData[i]]
	}
	scatter := unsafe(plotter.NewScatter(xys))

	scatter.GlyphStyleFunc = func(i int) draw.GlyphStyle {
		return draw.GlyphStyle{
			Color:  colors[i],
			Radius: vg.Points(3),
			Shape:  draw.CircleGlyph{},
		}
	}

	p.Add(scatter)

	wt := unsafe(p.WriterTo(300, 300, "png"))
	f := unsafe(os.Create(path))
	defer f.Close()

	wt.WriteTo(f)
}

var CMap map[float64]color.Color = map[float64]color.Color{
	0: color.RGBA{0, 0, 255, 255},
	1: color.RGBA{255, 0, 0, 255},
	2: color.RGBA{0, 255, 0, 255},
}
