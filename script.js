"use strict";
const form = document.querySelector(".form");
const fileInput = document.querySelector(".file");
const imagePreview = document.querySelector(".image-preview");
const attnPreview = document.querySelector(".attn-map-preview");
const info = document.querySelector(".info");
const loadingElement = document.getElementById("loading");
const resultsContainer = document.getElementById("results");

const aboutBtn = document.getElementById("about");
const aboutSection = document.getElementById("about-section");

aboutBtn.addEventListener("click", function () {
  if (aboutSection.classList.contains("hide")) {
    aboutSection.classList.remove("hide");
  } else {
    aboutSection.classList.add("hide");
  }
});

// Show image preview when file is selected
fileInput.addEventListener("change", function () {
  const file = this.files[0];
  if (file) {
    // remove
    attnPreview.style.display = "none";
    resultsContainer.innerHTML = "";

    document.getElementById("file-name").textContent = file.name;
    const reader = new FileReader();
    reader.addEventListener("load", function () {
      imagePreview.style.display = "block";
      imagePreview.innerHTML = `<img src="${this.result}" alt="Preview">`;
    });
    reader.readAsDataURL(file);
  } else {
    document.getElementById("file-name").textContent = "no file selected";
  }
});

// fetch data from api
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const imageFile = fileInput.files[0];
  if (!imageFile) {
    const message = "Please select a file first";
    document.querySelector(".info").textContent = message;
    return;
  }

  loadingElement.style.display = "flex";

  const formData = new FormData();
  formData.append("file", imageFile);

  try {
    const response = await fetch(
      "https://bamswastaken-kidney-detr-datican.hf.space/datican",
      {
        method: "POST",
        body: formData,
      }
    );

    if (!response.ok) {
      document.querySelector(".info").textContent = "an error occured";

      throw new Error("A problem occurred");
    }
    const data = await response.json();

    loadingElement.style.display = "none"; // remove loading animation

    console.log("API Response:", data);
    document.querySelector(".info").style.display = "none";
    processAndDisplayResults(imageFile, data);
  } catch (error) {
    console.error("Error:", error);
    loadingElement.style.display = "none";
    info.style.display = "inline";
    info.style.color = "red";
    info.textContent = "Check your Internet Connection!";
  }
});

// Add this function to process and display results
function processAndDisplayResults(imageFile, data) {
  if (!data.img_with_bboxes || !data.cross_attention_map) {
    info.style.display = "inline";
    info.style.color = "red";
    attnPreview.innerHTML = "";
    resultsContainer.innerHTML = "";

    info.textContent = "No kidney stone detected";
    return;
  }

  // Clear previous results
  resultsContainer.innerHTML = "";
  imagePreview.innerHTML = "";
  attnPreview.innerHTML = "";

  // SAFER: Create bbbx-img element properly
  const bboxImg = document.createElement("img");
  bboxImg.src = `data:image/png;base64,${data.img_with_bboxes}`; // Set source safely
  bboxImg.alt = "Detection Preview"; // Accessibility
  imagePreview.appendChild(bboxImg); // Append to DOM
  imagePreview.style.display = "inline";

  // SAFER: Create attn-img element properly
  const attnImg = document.createElement("img");
  attnImg.src = `data:image/png;base64,${data.cross_attention_map}`; // Set source safely
  attnImg.alt = "Attention Preview"; // Accessibility
  attnPreview.appendChild(attnImg); // Append to DOM
  attnPreview.style.display = "inline";

  // Create results table
  const table = document.createElement("table");
  table.className = "results-table";
  table.innerHTML = `
      <thead>
        <tr>
          <th>Class</th>
          <th>Confidence</th>
          <th>Bounding Box</th>
        </tr>
      </thead>
      <tbody>
      ${data.bboxes.boxes
        .map(
          (box, index) => `
        <tr>
          <td>${data.bboxes.labels[index] ?? "Unknown"}</td>
          <td>${(data.bboxes.scores[index] * 100).toFixed(1)}%</td>
          <td>X1:${box[0].toFixed(1)}, Y1:${box[1].toFixed(1)}<br>
              X2:${box[2].toFixed(1)}, Y2:${box[3].toFixed(1)}</td>
        </tr>
      `
        )
        .join("")}
    </tbody>
    `;
  resultsContainer.appendChild(table);
}
