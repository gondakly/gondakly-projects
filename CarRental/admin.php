<?php
// Include the database connection file
include 'db_connection.php'; // Ensure this path is correct

// Handle Delete
if (isset($_POST['delete_row'])) {
    $id = intval($_POST['row_id']);
    $table = $conn->real_escape_string($_POST['table_name']);
    $delete_sql = "DELETE FROM `$table` WHERE id = $id";
    if (!$conn->query($delete_sql)) {
        echo "<p>Error deleting record: " . $conn->error . "</p>";
    }
}

// Handle Insert
if (isset($_POST['insert_row'])) {
    $table = $conn->real_escape_string($_POST['table_name']);
    $columns_sql = "SHOW COLUMNS FROM `$table`";
    $columns_result = $conn->query($columns_sql);
    $columns = [];
    while ($row = $columns_result->fetch_assoc()) {
        if ($row['Field'] != 'id') { // Skip auto-increment ID
            $columns[] = $row['Field'];
        }
    }

    $values = [];
    foreach ($columns as $column) {
        $values[] = "'" . $conn->real_escape_string($_POST[$column]) . "'";
    }

    $columns_list = implode(", ", $columns);
    $values_list = implode(", ", $values);

    $insert_sql = "INSERT INTO `$table` ($columns_list) VALUES ($values_list)";
    if (!$conn->query($insert_sql)) {
        echo "<p>Error inserting record: " . $conn->error . "</p>";
    }
}

// Fetch tables
$tables_result = $conn->query("SHOW TABLES");
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="admin style.css">
    <title>Admin Panel</title>
</head>
<body>
    <div class="container">
        <h1>Admin Panel</h1>
        <?php while ($table_row = $tables_result->fetch_array()) {
            $table_name = $table_row[0];
            $rows_result = $conn->query("SELECT * FROM `$table_name`");
            $columns_result = $conn->query("SHOW COLUMNS FROM `$table_name`");
        ?>
        <h2><?php echo ucfirst($table_name); ?></h2>
        <table>
            <thead>
                <tr>
                    <?php while ($column = $columns_result->fetch_assoc()) { ?>
                        <th><?php echo $column['Field']; ?></th>
                    <?php } ?>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                <?php while ($row = $rows_result->fetch_assoc()) { ?>
                <tr>
                    <?php foreach ($row as $key => $value) { ?>
                        <td><?php echo htmlspecialchars($value); ?></td>
                    <?php } ?>
                    <td>
                        <form method="POST" style="display:inline-block;">
                            <input type="hidden" name="row_id" value="<?php echo $row['id']; ?>">
                            <input type="hidden" name="table_name" value="<?php echo $table_name; ?>">
                            <button type="submit" name="delete_row" class="btn-delete">Delete</button>
                        </form>
                    </td>
                </tr>
                <?php } ?>
            </tbody>
        </table>
        <form method="POST" class="insert-form">
            <h3>Insert New Row</h3>
            <?php
            $columns_result->data_seek(0); // Reset columns result pointer
            while ($column = $columns_result->fetch_assoc()) {
                if ($column['Field'] != 'id') { // Skip auto-increment ID
            ?>
            <input type="text" name="<?php echo $column['Field']; ?>" placeholder="<?php echo $column['Field']; ?>" required>
            <?php } } ?>
            <input type="hidden" name="table_name" value="<?php echo $table_name; ?>">
            <button type="submit" name="insert_row" class="btn-insert">Insert</button>
        </form>
        <?php } ?>
    </div>
</body>
</html>
