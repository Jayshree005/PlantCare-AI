
function getCookie(name) {
    const cookieValue = `; ${document.cookie}`;
    const parts = cookieValue.split(`; ${name}=`);

    if (parts.length === 2) {
        return parts.pop().split(";").shift();
    }

    return null;
}

// ======================================
// PLANTCARE AI - MAIN JAVASCRIPT
// ======================================

console.log("PlantCare AI JavaScript loaded!");

const API_BASE = "http://127.0.0.1:8000/api";

// ======================================
// PROTECT PAGES - LOGIN REQUIRED
// ======================================

const currentPage = window.location.pathname
    .split("/")
    .pop()
    .toLowerCase();

const protectedPages = [
    "upload.html",
    "history.html",
    "result.html"
];

const loggedInUser = localStorage.getItem("plantcareAuthUser");



if (protectedPages.includes(currentPage) && !loggedInUser) {
    console.log("PROTECTED PAGE CHECK:", currentPage, loggedInUser);

    alert("Please login first to access this page.");
    window.location.replace("login.html");
} else {
    console.log("PAGE ACCESS ALLOWED:", currentPage);
}

// ======================================
// 1. PLANT IMAGE UPLOAD ELEMENTS
// ======================================

const imageInput = document.getElementById("leaf-image");
const imagePreview = document.getElementById("image-preview");
const analyzeButton = document.getElementById("analyze-button");
const plantType = document.getElementById("plant-type");


// ======================================
// 2. IMAGE PREVIEW
// ======================================

if (imageInput && imagePreview && analyzeButton) {

    imageInput.addEventListener("change", function () {

        const file = imageInput.files[0];

        if (!file) {
            imagePreview.innerHTML = "";
            return;
        }

        if (!file.type.startsWith("image/")) {

            imagePreview.textContent =
                "Please select a valid image file.";

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


    // ======================================
    // 3. SEND IMAGE TO DJANGO + CNN MODEL
    // ======================================

    analyzeButton.addEventListener("click", async function () {

        const file = imageInput.files[0];

        // Check image selection
        if (!file) {

            imagePreview.textContent =
                "Please upload a leaf image first.";

            return;
        }

        // Check image format
        if (!file.type.startsWith("image/")) {

            imagePreview.textContent =
                "Please select a valid image file.";

            return;
        }

        const originalButtonText = analyzeButton.textContent;

        analyzeButton.disabled = true;
        analyzeButton.textContent = "Analyzing image...";

const formData = new FormData();

// Get selected plant
const selectedPlant = plantType ? plantType.value : "aloe";

// Send image and plant type to Django backend
formData.append("image", file);
formData.append("plant", selectedPlant);

console.log("Selected Plant:", selectedPlant);

        try {

            // Get CSRF token from Django
            await fetch("http://127.0.0.1:8000/api/csrf/", {
            method: "GET",
            credentials: "include"
        });

            const csrfToken = getCookie("csrftoken");

            console.log("Sending image to Django API...");

            // Send image to Django backend
            const response = await fetch(
                "http://127.0.0.1:8000/api/upload/",
                {
                    method: "POST",
                    credentials: "include",
                    headers: {
                    "X-CSRFToken": csrfToken
                    },
                body: formData
                }
            );

            const data = await response.json();

            console.log("Django API Response:", data);

            // Check API response
            if (!response.ok) {

                throw new Error(
                    data.error || "Image analysis failed."
                );

            }

            // Check CNN prediction
            if (!data.prediction) {

                throw new Error(
                    "CNN prediction was not returned by the backend."
                );

            }

            // Check confidence
            const confidence = data.confidence;

            // Convert image to Base64
            const imageData = await new Promise(
                function (resolve, reject) {

                    const reader = new FileReader();

                    reader.onload = function (event) {
                        resolve(event.target.result);
                    };

                    reader.onerror = function () {
                        reject(
                            new Error("Could not read the uploaded image.")
                        );
                    };

                    reader.readAsDataURL(file);

                }
            );

            // Save uploaded image
            localStorage.setItem(
                "uploadedLeafImage",
                imageData
            );

            // Save CNN prediction
            localStorage.setItem(
                "predictionName",
                data.prediction
            );

            // Save selected plant for result page
localStorage.setItem(
    "predictionPlant",
    selectedPlant
);

            // Format confidence
            let confidenceText = "--";

            if (confidence !== undefined && confidence !== null) {

                confidenceText = String(confidence);

                if (!confidenceText.endsWith("%")) {
                    confidenceText += "%";
                }

            }

            // Save confidence
            localStorage.setItem(
                "predictionConfidence",
                confidenceText
            );

            console.log("Prediction:", data.prediction);
            console.log("Confidence:", confidenceText);

            // Show success message briefly
            imagePreview.innerHTML = `
                <img src="${imageData}"
                     alt="Uploaded Plant Leaf">

                <p style="color: lightgreen; margin-top: 12px;">
                    Image analyzed successfully!
                </p>

                <p>
                    Predicted class: ${data.prediction}
                </p>

                <p>
                    Confidence: ${confidenceText}
                </p>
            `;

            // Redirect to result page
            window.location.href = "./result.html";


        } catch (error) {

            console.error("Image analysis error:", error);

            const errorMessage = document.createElement("p");

            errorMessage.style.color = "#ff8b8b";
            errorMessage.style.marginTop = "12px";
            errorMessage.textContent = error.message;

            imagePreview.appendChild(errorMessage);

        } finally {

            analyzeButton.disabled = false;
            analyzeButton.textContent = originalButtonText;

        }

    });

}

// DISPLAY LOGGED-IN USER NAME

const welcomeUser = document.getElementById("welcome-user");

if (welcomeUser) {
    const savedUser = localStorage.getItem("plantcareAuthUser");

    if (savedUser) {
        const user = JSON.parse(savedUser);

        const userName = user.name || "User";

        welcomeUser.textContent =
            "Hi, " +
            userName.charAt(0).toUpperCase() +
            userName.slice(1) +
            "!";
    }
}

// NAVBAR LOGIN / LOGOUT TOGGLE
(function updateNavbarAuth() {
    const authUser = localStorage.getItem("plantcareAuthUser");
    const loginLinks = document.querySelectorAll('a[href$="login.html"]');
    const signupLinks = document.querySelectorAll('a[href$="signup.html"]');
    const logoutBtn = document.getElementById("logout-button");

    if (authUser) {
        loginLinks.forEach(link => link.style.display = "none");
        signupLinks.forEach(link => link.style.display = "none");
        if (logoutBtn) logoutBtn.style.display = "inline-block";
    } else {
        loginLinks.forEach(link => link.style.display = "");
        signupLinks.forEach(link => link.style.display = "");
        if (logoutBtn) logoutBtn.style.display = "none";
    }
})();
// ======================================
// 4. RESULT PAGE - DISPLAY UPLOADED IMAGE
// ======================================

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


// ======================================
// 5. RESULT PAGE - DISPLAY CNN PREDICTION
// ======================================

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
        localStorage.getItem("predictionConfidence") ||
        "--";

}


// ======================================
// 6. PASSWORD VISIBILITY
// ======================================

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


// ======================================
// 7. LOGIN - FRONTEND DEMO
// ======================================



async function loginUser() {

    console.log("LOGIN BUTTON CLICKED!");

    const email =
        document.getElementById("login-email").value.trim();

    const password =
        document.getElementById("login-password").value;

    if (!email || !password) {
        alert("Please enter email and password.");
        return;
    }

    try {

        // Get CSRF token from Django
        await fetch(`${API_BASE}/csrf/`, {
            method: "GET",
            credentials: "include"
        });

        const csrfToken = getCookie("csrftoken");

        if (!csrfToken) {
            throw new Error("CSRF token not received from Django.");
        }

        // Login request
        const response = await fetch(`${API_BASE}/login/`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": csrfToken
            },
            credentials: "include",
            body: JSON.stringify({ email, password })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error ||
                data.detail ||
                data.message ||
                "Login failed."
            );
        }

        localStorage.setItem(
            "plantcareAuthUser",
            JSON.stringify(data.user || {})
        );

        console.log("Saved user:", localStorage.getItem("plantcareAuthUser"));

        console.log("Redirecting to upload page...");


       window.location.href = "./upload.html";

    } catch (error) {

        console.error("Login error:", error);
        alert(error.message || "Unable to login right now.");

    }
}




// ======================================
// 8. SIGNUP PASSWORD VALIDATION
// ======================================

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

        confirmPasswordInput.classList.add(
            "password-invalid"
        );

        confirmPasswordInput.classList.remove(
            "password-valid"
        );

        passwordError.textContent =
            "Passwords do not match.";

    } else {

        confirmPasswordInput.classList.remove(
            "password-invalid"
        );

        confirmPasswordInput.classList.add(
            "password-valid"
        );

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


// ======================================
// 9. SIGNUP - FRONTEND DEMO
// ======================================

async function signupUser() {

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
        /^(?=.*[^A-Za-z0-9\s]).{8,}$/;

    if (!passwordPattern.test(password)) {
        alert(
            "Password must contain at least 8 characters " +
            "and 1 special character."
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

    try {
        const csrfResponse = await fetch(`${API_BASE}/csrf/`, {
            method: "GET",
            credentials: "include"
        });

        if (!csrfResponse.ok) {
            throw new Error("Unable to retrieve a CSRF token from Django.");
        }

        const csrfToken = getCookie("csrftoken");

        if (!csrfToken) {
            throw new Error("CSRF token not received from Django.");
        }

        const response = await fetch(`${API_BASE}/signup/`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": csrfToken
            },
            credentials: "include",
            body: JSON.stringify({ name, email, password })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Signup failed.");
        }

        alert("Account created successfully. Please login.");
        window.location.href = "login.html";

    } catch (error) {
        console.error("Signup error:", error);
        alert(error.message || "Unable to create the account right now.");
    }

}
// LOGOUT FUNCTION
async function logoutUser() {
    try {
        const csrfResponse = await fetch(`${API_BASE}/csrf/`, {
            method: "GET",
            credentials: "include"
        });

        if (!csrfResponse.ok) {
            throw new Error("Unable to retrieve a CSRF token from Django.");
        }

        const csrfToken = getCookie("csrftoken");

        if (!csrfToken) {
            throw new Error("CSRF token not received from Django.");
        }

        const response = await fetch(`${API_BASE}/logout/`, {
            method: "POST",
            headers: {
                "X-CSRFToken": csrfToken
            },
            credentials: "include"
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error ||
                data.detail ||
                data.message ||
                "Logout failed."
            );
        }

        localStorage.removeItem("plantcareAuthUser");
        localStorage.removeItem("uploadedLeafImage");
        localStorage.removeItem("predictionName");
        localStorage.removeItem("predictionConfidence");

        alert(data.message || "You have been logged out successfully.");
        window.location.href = "login.html";
    } catch (error) {
        console.error("Logout error:", error);
        alert(error.message || "Unable to log out right now.");
    }
}