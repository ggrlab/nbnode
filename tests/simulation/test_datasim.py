from unittest import TestCase
import shutil
import os

from nbnode.testutil.datagen.simulate_odds_data import simulate_odds_data


class Test(TestCase):
    def test_simulate_odds_data(self):
        tmp = simulate_odds_data(n_samples_per_dist=3, n_points_per_sample=100)
        # rows: 3 samples, two classes, cols: 3 distributions + dist0_perc
        assert tmp.shape == (6, 4)

    def test_simulate_odds_data_newdir(self):
        tmpdir = "removeme_test_simulate_odds_data_newdir"
        shutil.rmtree(tmpdir, ignore_errors=True)
        tmp = simulate_odds_data(
            n_samples_per_dist=3, n_points_per_sample=100, target_data_dir=tmpdir
        )
        # rows: 3 samples, two classes, cols: 3 distributions + dist0_perc
        assert tmp.shape == (6, 4)
        assert len(os.listdir(tmpdir)) == 2
        assert len(os.listdir(os.path.join(tmpdir, "p_0"))) == 3
        assert len(os.listdir(os.path.join(tmpdir, "p_0.2"))) == 3

    def test_simulate_odds_data_csv_fcs(self):
        tmpdir = "removeme_test_simulate_odds_data_newdir"
        shutil.rmtree(tmpdir, ignore_errors=True)
        tmp = simulate_odds_data(
            n_samples_per_dist=3,
            n_points_per_sample=100,
            target_data_dir=tmpdir,
            savetype="fcs",
        )
        assert tmp.shape == (6, 4)
        assert len(os.listdir(tmpdir)) == 2
        assert len(os.listdir(os.path.join(tmpdir, "p_0"))) == 3
        assert len(os.listdir(os.path.join(tmpdir, "p_0.2"))) == 3
        tmp = simulate_odds_data(
            n_samples_per_dist=3,
            n_points_per_sample=100,
            target_data_dir=tmpdir,
            savetype="csv",
        )
        assert tmp.shape == (6, 4)
        assert len(os.listdir(tmpdir)) == 2
        # now there must be 6 files in each class as ".csv" and ".fcs" should be there
        assert len(os.listdir(os.path.join(tmpdir, "p_0"))) == 6
        assert len(os.listdir(os.path.join(tmpdir, "p_0.2"))) == 6
