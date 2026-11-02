
// ===============================
// PLANT IMAGE UPLOAD
// ===============================
console.log("PlantCare AI JavaScript loaded!");
const imageInput = document.getElementById("leaf-image");
const imagePreview = document.getElementById("image-preview");
const analyzeButton = document.getElementById("analyze-button");

// ===============================
// IMAGE PREVIEW
// ===============================

if (imageInput && imagePreview && analyzeButton) {

    imageInput.addEventListener("change", function () {
        const file = imageInput.files[0];

        console.log("Selected image:", file);

        if (!file) return;

        if (!file.type.startsWith("image/")) {
            imagePreview.innerHTML =
                "<span>Please select a valid image file.</span>";

            imageInput.value = "";
            return;
        }

        const reader = new FileReader();

        reader.onload = function (event) {
            imagePreview.innerHTML = `
                <img src="${event.target.result}"
                     alt="Selected Plant Leaf">
            `;
        };

        reader.readAsDataURL(file);
    });

    // ===============================
    // SEND IMAGE TO DJANGO + CNN
    // ===============================


    analyzeButton.addEventListener("click", async function () {

    console.log("ANALYZE BUTTON CLICKED");

    const file = imageInput.files[0];


    
        if (!file) {
            imagePreview.innerHTML =
                "<span>Please upload a leaf image first.</span>";
            return;
        }

        if (!file.type.startsWith("image/")) {
            imagePreview.innerHTML =
                "<span>Please select a valid image file.</span>";
            return;
        }

        const originalButtonText = analyzeButton.textContent;

        analyzeButton.disabled = true;
        analyzeButton.textContent = "Analyzing image...";

        const formData = new FormData();
        formData.append("image", file);

try {
    console.log("Sending image to Django API...");

    const response = await fetch(
        "http://127.0.0.1:8000/api/upload/",
        {
            method: "POST",
            body: formData
        }
    );

    console.log("HTTP status:", response.status);

    const data = await response.json();

    console.log("Django response:", data);
    
            if (!response.ok) {
                throw new Error(
                    data.error || "Image analysis failed."
                );
            }

            if (!data.prediction) {
                throw new Error(
                    "CNN prediction was not returned by the backend."
                );
            }

            const reader = new FileReader();

            reader.onload = function (event) {

                // Save uploaded image
                localStorage.setItem(
                    "uploadedLeafImage",
                    event.target.result
                );

                // Save CNN prediction
                localStorage.setItem(
                    "predictionName",
                    data.prediction
                );

                // Save model confidence
                localStorage.setItem(
                    "predictionConfidence",
                    data.confidence !== undefined
                        ? data.confidence + "%"
                        : "--"
                );

                // Show prediction on upload page
                imagePreview.innerHTML = `
                    <img src="${event.target.result}"
                         alt="Uploaded Plant Leaf">

                    <p style="color: lightgreen; margin-top: 12px;">
                        Image analyzed successfully!
                    </p>

                    <p>
                        Predicted class: ${data.prediction}
                    </p>

                    <p>
                        Confidence: ${
                            data.confidence !== undefined
                                ? data.confidence + "%"
                                : "--"
                        }
                    </p>
                `;

                // Navigate to result page
                // Navigate to result page
                console.log("Navigation code reached!");

                window.location.href = "./result.html";

            };
                

            reader.onerror = function () {
                imagePreview.innerHTML +=
                    "<p>Could not read the uploaded image.</p>";
            };

            reader.readAsDataURL(file);

        } catch (error) {
            imagePreview.innerHTML += `
                <p style="color: #ff8b8b; margin-top: 12px;">
                    ${error.message}
                </p>

                <p>
                    Please check that the Django server is running.
                </p>
            `;

        } finally {
            analyzeButton.disabled = false;
            analyzeButton.textContent = originalButtonText;
        }
    });
}


// ===============================
// RESULT PAGE: DISPLAY UPLOADED IMAGE
// ===============================

const resultLeafImage =
    document.getElementById("result-leaf-image");

if (resultLeafImage) {
    const uploadedImage =
        localStorage.getItem("uploadedLeafImage");

    if (uploadedImage) {
        resultLeafImage.src = uploadedImage;
    } else {
        resultLeafImage.alt = "No image uploaded";
    }
}


// ===============================
// RESULT PAGE: DISPLAY CNN PREDICTION
// ===============================

const predictionName =
    document.getElementById("prediction-name");

const predictionConfidence =
    document.getElementById("prediction-confidence");

if (predictionName) {
    predictionName.textContent =
        localStorage.getItem("predictionName") ||
        "No prediction available";
}

if (predictionConfidence) {
    predictionConfidence.textContent =
        localStorage.getItem("predictionConfidence") || "--";
}


// ===============================
// PASSWORD VISIBILITY
// ===============================

function togglePassword(inputId, button) {
    const passwordInput =
        document.getElementById(inputId);

    if (!passwordInput) return;

    if (passwordInput.type === "password") {
        passwordInput.type = "text";
        button.textContent = "🙈";
    } else {
        passwordInput.type = "password";
        button.textContent = "👁";
    }
}


// ===============================
// LOGIN - FRONTEND DEMO
// ===============================

function loginUser() {
    const email =
        document.getElementById("login-email").value.trim();

    const password =
        document.getElementById("login-password").value;

    if (!email || !password) {
        alert("Please enter email and password.");
        return;
    }

    // Real authentication will be connected later.
    window.location.href = "upload.html";
}


// ===============================
// SIGNUP PASSWORD VALIDATION
// ===============================

const signupPasswordInput =
    document.getElementById("signup-password");

const confirmPasswordInput =
    document.getElementById("confirm-password");

const passwordError =
    document.getElementById("password-error");

function checkPasswordMatch() {
    if (
        !signupPasswordInput ||
        !confirmPasswordInput ||
        !passwordError
    ) {
        return;
    }

    const password = signupPasswordInput.value;
    const confirmPassword = confirmPasswordInput.value;

    if (confirmPassword === "") {
        confirmPasswordInput.classList.remove(
            "password-invalid",
            "password-valid"
        );

        passwordError.textContent = "";
        return;
    }

    if (password !== confirmPassword) {
        confirmPasswordInput.classList.add("password-invalid");

        confirmPasswordInput.classList.remove("password-valid");

        passwordError.textContent = "Passwords do not match.";

    } else {
        confirmPasswordInput.classList.remove("password-invalid");

        confirmPasswordInput.classList.add("password-valid");

        passwordError.textContent = "";
    }
}

if (signupPasswordInput) {
    signupPasswordInput.addEventListener(
        "input",
        checkPasswordMatch
    );
}

if (confirmPasswordInput) {
    confirmPasswordInput.addEventListener(
        "input",
        checkPasswordMatch
    );
}


// ===============================
// SIGNUP - FRONTEND DEMO
// ===============================

function signupUser() {
    const name =
        document.getElementById("signup-name").value.trim();

    const email =
        document.getElementById("signup-email").value.trim();

    const password =
        document.getElementById("signup-password").value;

    const confirmPassword =
        document.getElementById("confirm-password").value;

    const userType =
        document.getElementById("user-type").value;

    const terms =
        document.getElementById("terms").checked;

    if (!name || !email || !password || !confirmPassword) {
        alert("Please fill all required fields.");
        return;
    }

    const passwordPattern =
        /^(?=.*[A-Z])(?=.*[!@#$%&]).{8,}$/;

    if (!passwordPattern.test(password)) {
        alert(
            "Password must contain at least 8 characters, " +
            "1 uppercase letter and 1 special character."
        );
        return;
    }

    if (password !== confirmPassword) {
        checkPasswordMatch();
        alert("Passwords do not match.");
        return;
    }

    if (!userType) {
        alert("Please select your user type.");
        return;
    }

    if (!terms) {
        alert("Please agree to the Terms & Conditions.");
        return;
    }

    // Real account creation will be connected to Django later.
    alert(
        "Frontend validation successful. " +
        "Signup backend is not connected yet."
    );

    window.location.href = "login.html";
}