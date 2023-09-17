document.getElementById('imageInput').addEventListener('change', function () {
    const file = this.files[0];
    const reader = new FileReader();

    reader.onload = function () {
        const previewImage = document.getElementById('previewImage');
        previewImage.src = reader.result;
    }

    if (file) {
        reader.readAsDataURL(file);
    }
});
