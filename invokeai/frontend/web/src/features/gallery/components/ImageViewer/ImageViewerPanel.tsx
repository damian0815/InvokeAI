import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectImageToCompare, selectLastSelectedItem, selectTokenizationDisplayMode } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';

import { ImageViewerContextProvider } from './context';
import { ImageComparison } from './ImageComparison';
import { ImageTokenization } from './ImageTokenization';
import { ImageViewer } from './ImageViewer';
import { VideoViewer } from './VideoViewer';

const selectIsComparing = createSelector(
  [selectLastSelectedItem, selectImageToCompare],
  (lastSelectedImage, imageToCompare) => !!lastSelectedImage && !!imageToCompare
);

const selectIsTokenizing = createSelector([selectLastSelectedItem, selectTokenizationDisplayMode],
  (lastSelectedImage, tokenizationDisplayMode) => !!lastSelectedImage && !!tokenizationDisplayMode
);

export const ImageViewerPanel = memo(() => {
  const isComparing = useAppSelector(selectIsComparing);
  const isTokenizing = useAppSelector(selectIsTokenizing);
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);

  const notComparingOrTokenizing = !isComparing && !isTokenizing;

  return (
    <ImageViewerContextProvider>
      {
        // The image viewer renders progress images - if no image is selected, show the image viewer anyway
        notComparingOrTokenizing && !lastSelectedItem && <ImageViewer />
      }
      {notComparingOrTokenizing && lastSelectedItem?.type === 'image' && <ImageViewer />}
      {notComparingOrTokenizing && lastSelectedItem?.type === 'video' && <VideoViewer />}
      {isComparing && <ImageComparison />}
      {isTokenizing && <ImageTokenization />}
    </ImageViewerContextProvider>
  );
});
ImageViewerPanel.displayName = 'ImageViewerPanel';
