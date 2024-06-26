package me.tsvrn9.beginnerneuralnetwork;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.util.stream.IntStream;
import java.util.List;

import me.tsvrn9.beginnerneuralnetwork.neuralnetwork.NeuralNetwork;
import me.tsvrn9.beginnerneuralnetwork.neuralnetwork.NetworkVector;
import me.tsvrn9.beginnerneuralnetwork.neuralnetwork.functions.Functions;
import org.jblas.DoubleMatrix;
import me.tsvrn9.beginnerneuralnetwork.serialization.SimpleSerialization;

public class DrawingApp extends JFrame {

    private final int canvasWidth = 28;
    private final int canvasHeight = 28;
    private final int scale = 20; // Scale factor for rendering
    private final JLabel predictionLabel;
    private final JPanel drawingPanel;

    private final BufferedImage canvas;
    private final Graphics2D canvasGraphics;
    private final DoubleMatrix imageMatrix;
    private final boolean[][] drawingGrid;

    private final NeuralNetwork neuralNetwork;
    private boolean isDrawing;

    public DrawingApp() {
        canvas = new BufferedImage(canvasWidth, canvasHeight, BufferedImage.TYPE_INT_RGB);
        canvasGraphics = canvas.createGraphics();
        canvasGraphics.setColor(Color.WHITE);
        canvasGraphics.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());

        imageMatrix = new DoubleMatrix(canvasWidth*canvasHeight);
        drawingGrid = new boolean[canvasWidth][canvasHeight];

        neuralNetwork = new NeuralNetwork(
                SimpleSerialization.load(Config.FILE_PATH, NetworkVector.class),
                Functions.sigmoid, Functions.crossEntropy);

        drawingPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2d = (Graphics2D) g;
                g2d.scale(scale, scale); // Scale up the rendering
                g2d.drawImage(canvas, 0, 0, this);
            }
        };

        drawingPanel.setPreferredSize(new Dimension(canvas.getWidth() * scale, canvas.getHeight() * scale));

        drawingPanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                int x = e.getX() / scale; // Scale down the mouse coordinates
                int y = e.getY() / scale;

                if (isValidPixel(x, y)) {
                    draw(x, y, SwingUtilities.isLeftMouseButton(e));
                }

                isDrawing = true;
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                isDrawing = false;
            }
        });

        drawingPanel.addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                if (isDrawing) {
                    int x = e.getX() / scale; // Scale down the mouse coordinates
                    int y = e.getY() / scale;

                    for (int a = -1; a <= 1; a++) {
                        for (int b = -1; b <= 1; b++) {
                            if (isValidPixel(x + a, y + b)) {
                                draw(x + a, y + b, SwingUtilities.isLeftMouseButton(e));
                            }
                        }
                    }
                }
            }
        });

        JButton clearButton = new JButton("Clear");
        clearButton.addActionListener(e -> {
            clearCanvas();
            drawingPanel.repaint();
            updateImageMatrix();
            predictValue();
        });

        predictionLabel = new JLabel("Prediction: ");
        predictionLabel.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 16));
        predictionLabel.setHorizontalAlignment(SwingConstants.CENTER);

        setLayout(new BorderLayout());
        add(drawingPanel, BorderLayout.CENTER);
        add(clearButton, BorderLayout.SOUTH);
        add(predictionLabel, BorderLayout.NORTH);

        setLayout(new BorderLayout());
        add(drawingPanel, BorderLayout.CENTER);
        add(clearButton, BorderLayout.SOUTH);
        add(predictionLabel, BorderLayout.NORTH);

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
        setLocationRelativeTo(null);
        setVisible(true);

        // MnistDataset dataset = Objects.requireNonNull(MnistReader.readData(me.tsvrn9.Config.TESTING_IMAGES, me.tsvrn9.Config.TESTING_LABELS));
        // int randomIndex = (int) (Math.random() * dataset.length());
        // drawImage(dataset.getData(randomIndex));
        // System.out.println(dataset.getLabel(randomIndex));
    }

    private void draw(int x, int y) {
        draw(x, y, true);
    }

    private void draw(int x, int y, boolean value) {
        drawingGrid[x][y] = value;
        canvasGraphics.setColor(value ? Color.BLACK : Color.WHITE);
        canvasGraphics.fillRect(x, y, 1, 1);
        drawingPanel.repaint();

        updateImageMatrix();
        predictValue();
    }

    private void clearCanvas() {
        canvasGraphics.setColor(Color.WHITE);
        canvasGraphics.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());

        for (int x = 0; x < canvasWidth; x++) {
            for (int y = 0; y < canvasHeight; y++) {
                drawingGrid[x][y] = false;
            }
        }
    }

    private boolean isValidPixel(int x, int y) {
        return x >= 0 && x < canvasWidth && y >= 0 && y < canvasHeight;
    }

    private void updateImageMatrix() {
        if (Config.CENTER_DRAWING)
            updateImageMatrixCentered();
        else {
            for (int x = 0; x < canvasWidth; x++) {
                for (int y = 0; y < canvasHeight; y++) {
                    imageMatrix.put(y + canvasHeight*x, drawingGrid[x][y] ? 1.0 : 0.0);
                }
            }
        }
    }

    private void updateImageMatrixCentered() {
        // Calculate the center coordinates
        int centerX = canvasWidth / 2;
        int centerY = canvasHeight / 2;

        // Find the bounding box of the non-blank pixels in the image
        int minX = canvasWidth;
        int minY = canvasHeight;
        int maxX = 0;
        int maxY = 0;

        for (int y = 0; y < canvasHeight; y++) {
            for (int x = 0; x < canvasHeight; x++) {
                if (drawingGrid[y][x]) {
                    minX = Math.min(minX, x);
                    minY = Math.min(minY, y);
                    maxX = Math.max(maxX, x);
                    maxY = Math.max(maxY, y);
                }
            }
        }

        // Calculate the shift amounts to center the image
        int shiftX = centerX - (maxX + minX) / 2;
        int shiftY = centerY - (maxY + minY) / 2;

        // Create a new array to hold the centered image
        double[][] centeredImage = new double[canvasHeight][canvasHeight];

        // Copy the pixels from the original image to the centered image with the calculated shift
        for (int y = 0; y < canvasHeight; y++) {
            for (int x = 0; x < canvasWidth; x++) {
                int shiftedX = x + shiftX;
                int shiftedY = y + shiftY;

                // Check if the shifted coordinates are within the bounds of the centered image
                if (shiftedX >= 0 && shiftedX < canvasWidth && shiftedY >= 0 && shiftedY < canvasHeight) {
                    centeredImage[shiftedY][shiftedX] = drawingGrid[y][x] ? 1.0 : 0.0;
                }
            }
        }
        for (int x = 0; x < canvasWidth; x++) {
            for (int y = 0; y < canvasHeight; y++) {
                imageMatrix.put(y + canvasHeight*x, centeredImage[x][y]);
            }
        }
    }

    private void predictValue() {
        DoubleMatrix y = neuralNetwork.input(imageMatrix);
        List<Integer> sorted = IntStream.range(0, 10)
                .boxed()
                .sorted((a, b) -> Double.compare(y.get(b), y.get(a)))
                .toList();
        List<Double> sortedConfidences = sorted.stream()
                .map(y::get)
                .map(d -> Math.round(d * 1000) / 1000.0)
                .toList();

        System.out.println();
        System.out.println("Prediction Matrix: " + y);
        System.out.println("Options ranked: " + sorted);
        System.out.println("Confidences: " + sortedConfidences);
        System.out.println();
        // Perform prediction and handle the result
        predictionLabel.setText("Prediction: " + y.argmax());
    }

    private void drawImage(DoubleMatrix image) {
        for (int x = 0; x < canvasWidth; x++) {
            for (int y = 0; y < canvasHeight; y++) {
                draw(x, y, image.get(y + x*canvasHeight) > 0.2);
            }
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(DrawingApp::new);
    }
}
