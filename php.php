<?php

	$longitude1 = $_GET['longitude1'];
	$latitude1 = $_GET['latitude1'];

	$longitude2 = $_GET['longitude2'];
	$latitude2 = $_GET['latitude2'];

	$longitude3 = $_GET['longitude3'];
	$latitude3 = $_GET['latitude3'];

	$station= $_GET['select_at'];

	$page = $_GET['page'];

	$rayon1 = $_GET['rayon1'];
	$rayon2 = $_GET['rayon2'];

if ($page == 'page1') {
	exec("python3 ./py/php.py $longitude1 $latitude1 $station $page ");
}

if ($page == 'page2') {
	exec("python3 ./py/php.py $longitude2 $latitude2 $rayon1 $page ");
}


if ($page == 'page3') {
	exec("python3 ./py/php.py $longitude3 $latitude3 $rayon2 $page ");
}

if (shell_exec("python3 ./py/main.py")) {
    echo 'OK';
} else {
    echo 'Not OK ';
}

?>

<!-- <?php header("Location: index.html");?> -->
