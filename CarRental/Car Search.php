<!DOCTYPE html>
<html lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search The Car</title>
    <link rel="stylesheet" href="Style reg Cars.css">
    <div class="form-box">
        <div class="top-section"><h1>Search the car</h1></div>
        <div class="form-content">
    <form action="search_cars.php" method="GET">

        <label for="model">No of Seats</label>
        <input type="text" id="Seats" name="Seats" placeholder="Ex:6 seats">

        <label for="year">Year</label>
        <input type="number" id="year" name="year" placeholder="ex:2020">

        <label for="color">Color</label>
        <input type="text" id="color" name="color" placeholder="ex:Red"">

        <label for="price">Price Per Hour:</label>
        <input type="number" id="price" name="price" placeholder="ex:200$">

        <button type="submit">Search</button>
    </form>
</div>

</body>
</html>