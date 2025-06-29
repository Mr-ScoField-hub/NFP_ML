fetch("predictions.json")
  .then((response) => response.json())
  .then((data) => {
    document.getElementById(
      "direction"
    ).innerText = `Direction: ${data.direction}`;
    document.getElementById(
      "magnitude"
    ).innerText = `Magnitude: ${data.magnitude}`;
    document.getElementById(
      "confidence"
    ).innerText = `Confidence: ${data.confidence}%`;
  })
  .catch((error) => console.error("Error fetching predictions:", error));
