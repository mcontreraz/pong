package main

import (
	"fmt"
	"image/color"
	"math/rand"
	"strconv"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

const (
	screenWidth      = 640
	screenHeight     = 480
	paddleWidth      = 16
	paddleHeight     = 80
	paddleSpeed      = 4
	ballSize         = 16
	ballSpeed        = 4
	initialBallSpeed = 4
)

type Game struct {
	playerY  float64
	ballX    float64
	ballY    float64
	ballDX   float64
	ballDY   float64
	score    int
	maxScore int
	reset    bool
}

func (g *Game) Update() error {
	if ebiten.IsKeyPressed(ebiten.KeyUp) {
		g.playerY -= paddleSpeed
	}
	if ebiten.IsKeyPressed(ebiten.KeyDown) {
		g.playerY += paddleSpeed
	}

	if g.playerY < 0 {
		g.playerY = 0
	}
	if g.playerY > screenHeight-paddleHeight {
		g.playerY = screenHeight - paddleHeight
	}

	if ebiten.IsKeyPressed(ebiten.KeyEnter) && g.reset {
		g.resetGame()
	}

	g.ballX += g.ballDX
	g.ballY += g.ballDY

	if g.ballY < 0 || g.ballY > screenHeight-ballSize {
		g.ballDY *= -1
	}

	if g.ballX < 0 {
		g.resetGame()
	}

	if g.ballX > screenWidth-ballSize {
		g.ballDX *= -1
	}

	if g.ballX < paddleWidth && g.ballY+ballSize > g.playerY && g.ballY < g.playerY+paddleHeight {
		g.ballDX *= -1
		g.ballDY = (g.ballY + ballSize/2 - (g.playerY + paddleHeight/2)) / (paddleHeight / 2) * initialBallSpeed
		g.score++

		// Agregar aleatoriedad a la dirección de la pelota
		g.ballDX += rand.Float64() - 0.5
		g.ballDY += rand.Float64() - 0.5

		// Actualizar el puntaje máximo si es necesario
		if g.score > g.maxScore {
			g.maxScore = g.score
		}
	}

	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	ebitenutil.DrawRect(screen, paddleWidth, g.playerY, paddleWidth, paddleHeight, color.White)
	ebitenutil.DrawRect(screen, g.ballX, g.ballY, ballSize, ballSize, color.White)
	ebitenutil.DebugPrint(screen, "Score: "+strconv.Itoa(g.score))
	ebitenutil.DebugPrintAt(screen, "Max Score: "+strconv.Itoa(g.maxScore), 0, 20)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Ball Y: %.2f", g.ballY), 0, 40)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Ball X: %.2f", g.ballX), 0, 60)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Paddle: %.2f", g.playerY), 0, 80)
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return screenWidth, screenHeight
}

func (g *Game) resetGame() {
	g.playerY = screenHeight / 2
	g.ballX = screenWidth / 2
	g.ballY = screenHeight / 2
	g.ballDX = ballSpeed
	g.ballDY = ballSpeed
	g.score = 0
	g.reset = false
}

func main() {
	fmt.Println("Starting the program...")
	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Pong")
	game := &Game{
		playerY: screenHeight / 2,
		ballX:   screenWidth / 2,
		ballY:   screenHeight / 2,
		ballDX:  ballSpeed,
		ballDY:  ballSpeed,
		reset:   true,
	}

	if err := ebiten.RunGame(game); err != nil {
		fmt.Println(err)
	}
}
